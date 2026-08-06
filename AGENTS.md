# AGENTS.md

本文件是 AI agent 在 vllm-ascend 仓库工作时的核心约束。

## 项目定位

vllm-ascend 是 vLLM 的昇腾 NPU 硬件插件。不直接加模型文件；模型相关功能通过 `vllm_ascend/patch/`（patch 上游）或继承（`NPUModelRunner` 等）实现。新增 patch / model_runner 行为需架构评审。

## 环境

- torch 2.10.0 + torch_npu 2.10.0，triton 3.2.0（CANN ascend 后端）
- 8 × Ascend910B4-1（单 die，20 AIC + 40 AIV，HBM 64GB @ 1600MHz）
- `msprof`：`/usr/local/Ascend/cann-9.0.0/bin/msprof`（需在 PATH）
- 仅用 `npu:0` 测试时设 `ASCEND_RT_VISIBLE_DEVICES=0`

### 910B4-1 硬件性能（SOL 基准）

数据来源：`/usr/local/Ascend/cann-9.0.0/aarch64-linux/data/platform_config/Ascend910B4-1.ini`（`cube_freq=1500`、`cube_m/n/k=16`、`DT_INT8=16,32,16`、`l2_size=96MB`）。

| 指标 | 值 | 计算 |
|------|-----|------|
| BF16/FP16 cube | **245.76 TFLOPS** | 20 AIC × 16×16×16 MAC × 2 × 1.5GHz |
| INT8 | **491.52 TOPS** | K=32，较 BF16 翻倍（INT4 再翻倍） |
| HBM 带宽 | **1.6 TB/s** | |
| L2 | 96 MB | 单 die 共享 |
| L1 / L0A / L0B / L0C / UB | 512KB / 64KB / 64KB / 128KB / 192KB | 每 AIC |
| 机器平衡点 | 153.6 FLOP/B | 245.76e12 / 1.6e12，低于此即 mem-bound |

**算子性能评价一律以这组峰值计算 SOL**（bench_deepseek_v4.py 中的 `P_CUBE_FLOPS=246e12`、`BW_HBM_BPS=1.6e12` 即来源于此），禁止用其它型号（910B3/Atlas A3 等）的峰值。

## 算子三类与启用方式

| 类别 | 调用 | 启用 |
|------|------|------|
| torch_npu（aclnn） | `torch_npu.npu_*` | `import torch_npu` |
| AscendC 自定义 | `torch.ops._C_ascend.*` | `from vllm_ascend.utils import enable_custom_op; enable_custom_op()` |
| Triton | `vllm_ascend/ops/triton/*` | `from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton; init_device_properties_triton()` |

## 编译与安装（自定义算子）

**唯一推荐入口**（setup.py 一站式驱动，不要手工拼各步）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=910b MAX_JOBS=256 pip install -e . --no-build-isolation --no-deps
```

- `--no-deps` 必须：离线环境 pip 会卡在依赖解析的网络重试上。
- 流程：setup.py → `csrc/build_aclnn.sh`（编 vendor 算子包并装到 `vllm_ascend/_cann_ops_custom`）→ cmake 编 `vllm_ascend_C*.so` / `libvllm_ascend_kernels.so` → 部署到源码树 `vllm_ascend/`。
- csrc/build 树是热的时算子包为增量编译（几分钟）；冷启动全量 30–60 min。

### 新增/修改算子

1. `csrc/<group>/<op>/` 下建 `op_kernel/` + `op_host/` + `CMakeLists.txt`（复制同类算子改）；算子名加入 `csrc/build_aclnn.sh` 对应 SOC 分支的 `CUSTOM_OPS_ARRAY`。
2. torch 注册三处：`csrc/torch_binding.cpp`（函数 + schema + impl）、`csrc/torch_binding_meta.cpp`（meta + impl）。
3. **host 侧 ODR 坑**：所有算子的 tiling .cpp 链接进同一个 `libcust_opmaster_rt2.0.so`，非 static 自由函数（如 `LayoutTypeToStr`）跨算子重名 → multiple definition。复制算子代码时必须连自由函数一起改名。
4. **kernel 源码拷贝坑**：op_kernel 源码被复制到 `csrc/build/binary/ascend910b/src/<op>/` 并打 `.done` 标记；改了 kernel 源码必须删该目录（或 `.done`），否则编译的还是旧拷贝。
5. **stale 缓存坑**：删算子/删源文件后必须清构建树——`AICPU_CUST_OBJ_TARGETS` 是 CMake `CACHE INTERNAL`，会累积已删除算子导致 generate 失败；`build/temp.*` 里的 `auto_gen/`、`vllm_ascend_kernels_merge_obj_dir/`、`*-prefix/` 同理。清不干净就整个删 `csrc/build` 和 `build/`。
6. **杀构建进程**：`pkill -f "build.sh"` 会匹配到自己 shell 的命令行（自杀），用字符类写法 `pkill -f "build[.]sh"`；杀完删 `csrc/build` 下的 `kernel_meta.lock` 残留，否则 opc 报 `Another process is using this dir`。
7. `.run` 安装器的 `--install-path` 必须是绝对路径。

### 单独重编（不跑 pip）

```bash
# vendor 算子包（ops 列表从 build_aclnn.sh 对应分支抄）
cd csrc && bash build.sh --pkg --ops="op1;op2;..." --soc="ascend910b" -j256
./build/cann-ops-transformer-custom_linux-aarch64.run --install-path=$PWD/../vllm_ascend/_cann_ops_custom  # 绝对路径
# 主扩展（torch_binding 改动）
cmake --build build/temp.linux-aarch64-cpython-311 -j256
cp build/temp.linux-aarch64-cpython-311/vllm_ascend_C*.so build/temp.linux-aarch64-cpython-311/lib/libvllm_ascend_kernels.so vllm_ascend/
```

### 验证

```bash
python -c "from vllm_ascend.utils import enable_custom_op; enable_custom_op(); import torch; print(hasattr(torch.ops._C_ascend, '<op_name>'))"
```

## NPU 性能硬规则

- **禁止在热路径用 `tensor.item()`**：device tensor 的 `.item()` 触发 NPU->CPU 同步，阻塞 AsyncScheduler。改用 device 侧算子（`torch.argmax`/`torch.sum`）或批量同步。
- 热路径禁止 CPU-NPU 内存搬运。
- 计时基准用 msprof 的 `Task Duration(us)`，**不要用 host `time.perf_counter`**（含 launch 开销，会失真）。

## Profiling 工具

- **msprof 测量必须独占 NPU**：卡上有其它负载时同输入 T_k 可漂移数倍且无规律（曾把 30us 测成 1.1ms）。测试前用 `npu-smi info` 确认卡空闲，用 `ASCEND_RT_VISIBLE_DEVICES=<空闲卡>` 指定。

- 算子基准与 msprof 封装：`benchmarks/ops_profiling/`（见其 `README.md`）
    - `msprof app` -> `op_summary_*.csv`（每 kernel 设备侧耗时 + AIC/AIV/MTE 分项）
    - `msprof op`（默认参数）-> `PipeUtilization.csv`（每 block 各 pipe 占用率）
- Triton IR dump：`TRITON_DEBUG=1 TRITON_DUMP_DIR=<dir>`，产出 `kernel.ttir/ttadapter/npuir.mlir`
- 评测方法论：`docs/昇腾 910B 硬件架构与 SOL 性能评测方法论.md`（理论）+ `docs/vllm-ascend 算子性能评测实践指南.md`（实践）。**算子性能评价以 SOL gap 为准，不是 speedup。**

## 环境变量

新增环境变量必须加到 `vllm_ascend/envs.py` 的 `env_variables` 字典并写文档，命名 `VLLM_ASCEND_*`，禁止在代码里硬编码 env 名。性能关键路径上的新 env 需评审。

## 测试与提交

- UT：`tests/ut/`，E2E：`tests/e2e/`。新功能必须带测试，bugfix 带回归测试。
- 提交前：`ruff check vllm_ascend/`、`ruff format vllm_ascend/`、`bash format.sh ci`（含 markdownlint）。
- Commit 用 Conventional Commits 且**必须 sign-off**（`git commit -s`）：`feat(npu): summary` + body。
- PR 从个人 fork 提交，标题 `[Type][Module] Description`，描述遵循 `.github/PULL_REQUEST_TEMPLATE.md`。

## 命名

类 `PascalCase`，函数/变量 `snake_case`，常量 `ALL_UPPER_CASE`。禁止 magic number，用命名常量。避免新增可变全局状态。
