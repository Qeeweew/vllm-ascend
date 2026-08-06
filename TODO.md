# TODO: compressor 拆分（MatMulV3 ×2 + compressor_epilogue）——现状与待解 bug

## 1. 背景与目标

DeepSeek-V4 DSA 的 `Compressor` 融合算子（`torch.ops._C_ascend.compressor`，csrc/attention/compressor）内部 = cube GEMM（x@wkv、x@wgate）+ AIV epilogue（ape/softmax/加权压缩/state 递归/rms_norm/rope）。910B4-1 实测（msprof，卡 1/7）：

- prefill M=8192：**T=1926us，仅 125 TFLOPS（50.8% cube）**；对照纯 MatMulV3 [8192,7168]×[7168,2048] = 1006us（**97.4% cube**）
- 瓶颈：cube 的 L1 复用差（`aic_mte2_ratio=78.8%`，GEMM 本身 mac_time≈950us 但被 MTE 拖倍），AIV epilogue 本身只有 ~70us
- decode B=1/8/64：47.6/54.6/78.0us（D=2.6/2.9/3.8）

**方案**：拆成 2 个 `F.linear`（MatMulV3，[M,7168]×[7168,1024] ×2，~94% cube）+ 新 AIV-only 算子 `compressor_epilogue` 做剩下的向量部分。预期 prefill M=8192：~520us(GEMM) + ~100-200us(epilogue) ≪ 1926us。

- 生产权重形状（vllm `deepseek_compressor.py` 确认）：`wkv/wgate` 各 **[1024, 7168]**（coff(2)×headDim(512)），`ape [cmpRatio, 1024] fp32`，`state_cache [blocks, 8, 2*coff*headDim] fp32`
- 生产调用（dsa_v1.py 4 处）全部：`rotary_mode=2, cache_mode=1`，x 2D（TH layout），bf16，rope fp32

## 2. 迁移方法（新算子如何来的）

`csrc/attention/compressor_epilogue/` = **当前仓库版** `csrc/attention/compressor/` 的 arch32 拷贝 + 全量改名（`Compressor→CompressorEpilogue`、`COMPRESSOR_→COMPRESSOR_EPILOGUE_`）+ 如下删改：

> ⚠️ 注意：仓库历史里曾有 compressor 重构提交（02b5b4791 等，语义不同：score=[T,4] 标量门控），已被 reset 掉。**必须以当前仓库版（OverLap/SoftmaxDN/KvMulReduceScore 结构）为基线**，第一版误用了重构版语义，全部作废重写。

### kernel 侧（op_kernel/）

1. **入口 `compressor_epilogue.cpp`**：参数 `mmKv/mmScore`（外部 GEMM 输出 bf16，[M,1024]）替换 `x/wKv/wGate`；arch32-only；`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only 但保留 SyncAll；`KERNEL_TYPE_AIV_ONLY` 不能用 SyncAll）
2. **driver `..._kernel_perf.h`**：删 cube 全部（block_cube include/成员、`ComputeMm1`、AIC 分支、cube↔vec CrossCore flag）；`aiCoreIdx=GetBlockIdx()`；workspace 只剩 vec1Res（+v1v2 double buffer）；`dbSize=coreGroupNum*vec1ResSize`
3. **vec 块 `..._block_vec_perf.h`**（唯一改数据流的地方）：
   - `FromWokrSpaceToUb`：从用户 mm GM 读（替代 cube 写的 fp32 workspace）。展平行偏移 = `(tools_.GetTIdxByBatch(sliceInfo.bIdx) + sliceInfo.sIdx) * coff*headDim + dStartIdx`（mm GM 行序与原 workspace 一致，都是 (batch,token) 展平、行内 [coff,headDim]）。bf16 暂存在 fp32 目标 buffer 后半段（字节不重叠）后 `Cast` 成 fp32
   - `LoadFromWorkSpace`：删掉 cacheTc 分支和 `SaveToWorkSpace`（原机制为跨基本块尾部行缓存；本算子 mm GM 全 call 持久，直接读）。`isFirst` 分支 = 一次 2D copy（≤cmpRatio 行）+ 一次 Cast + 一次 UB→UB 2D copy（tailStageBuf 2KB：前 1KB bf16 暂存、后 1KB fp32）
   - 其余（PadAlign/OverLap/SoftmaxDN/KvMulReduceScore/ReadState/SaveState/Vec2 全部）**逐字节与原算子一致**
4. **同步**：只保留 vec2 前的 `SyncAll()`（每 nSize=2 块一次）。原因：vec1 按 D 切 8 份（dBaseSize=64），一行压缩 token 的 8 个 d-chunk 由 8 个不同核写 vec1Res；vec2 的 rms_norm 要整行 512 维 → 必须等齐。cacheTc 删除后每块屏障（SYNC_V1_FLAG）已删

### host 侧（op_host/）

- def：输入 `mm_kv/mm_score`(bf16|fp16)，输出 `cmp_kv` + `state_cache`（in-place），attr 同原算子
- proto：输入重编号（0..10），输出 dim0 推断逻辑不变
- tiling：workspace 只算 vec1Res（`mBase*headDim*nSize*coreGroupNum*4B*dbRatio`）；`usedCoreNum=aivNum_/2=20`（**必须保持 MIX 语义：vec 代码内部 `aiCoreNum = usedCoreNum * 2` = 40**，之前误设 40 导致 80 核错配置）；`blockDim=aivNum_=40`；`DAY0_SCOPE` 宏已加
- **tiling key 裁剪**（两个算子的 `template_tiling_key.h`）：SEL 192→4 key = {TH, BF16, COFF∈{1,2}, ROTARY=2, CACHE=1, TEMPLATE∈{EMPTY_X=1,PERF=2}, ROPE=FP32}。编译时间 14min→4.5min。非生产组合运行时会在 tiling key 匹配报错，需要时放开

### 注册与 Python

- `csrc/torch_binding.cpp` / `torch_binding_meta.cpp`：`compressor_epilogue` schema + impl（state_cache 为 `Tensor(a!)`）
- `vllm_ascend/envs.py`：`VLLM_ASCEND_DSA_COMPRESSOR_SPLIT`（默认 "0"）
- `vllm_ascend/attention/dsa_v1.py`：`_run_compressor`（split: 2×`F.linear` + epilogue；fallback 融合），4 个调用点已接
- bench builder `bench_deepseek_v4.py::_build_dsa_compressor` 支持 split（env 切换）

## 3. 已验证的语义（torch 参考对上 fused）

压缩 token k 的窗口 = **8 行 = [前一组 4 token | 当前组 4 token]**（coff=2 overlap）：
- mm 行内布局 `[coff0(512) | coff1(512)]`；**当前组用 coff1 块，前一组用 coff0 块**（UB 内 D_L/D_R）
- `score' = mm_score + ape`（ape 按窗口内位置行加）；**逐通道**对 8 行做 softmax（ColumnSoftMax）；`cmp[c] = Σ₈ p·kv`（ColumnSum）
- `SaveState` 把 score'/kv 按绝对 seq 下标写进 state_cache（分页，供 sparse_attn 用）；`ReadState` 只在 call 边界（bStartPos>0 的 chunk/decode 续算）读**上一个 call** 的行
- vec1Res → vec2：rms_norm（整行 512）+ rope（后 64 维）→ cmp_kv bf16
- 验证：wgate=0、ape=0 时 fused 输出 = 4 行 coff1 的均值（torch 参考差 0.018，bf16 舍入量级）✓

## 4. Bug 已修复（2026-08-06）

四个根因，全部在 `compressor_epilogue_block_vec_perf.h`：

1. **bf16 数据走 fp32 的块长换算**（`DataCopyAlign*` 系列 helper）：`DataCopyParams.blockLen/srcGap/dstGap` 单位是 32B 块，原代码硬编码 `/FP32_BLOCK_ELEMENT_NUM`（8 元素/块，fp32 语义）；split 的 mm GM 是 X_T（bf16/fp16，16 元素/块）→ 每行多拷一倍、行推进 1024 元素（整整一个 token）→ coff0/coff1 交错行全错。修复：helper 改为按 `BYTE_BLOCK/sizeof(O)` 换算（fp32 调用点数值不变）。
2. **`LoadFromWorkSpace` 的 GM 行 stride 抄错**：原算子读前驱 token 的 coff0 半边（窗口 D_L），行 stride = `coff_*headDim`(1024)；第一版误用 `FromWokrSpaceToUb` 的交错 stride `headDim`(512) → dDealSize=512 时左半窗口读成交错垃圾。表现为每核第一个 tc 错、第二个对（prefill 65% 错）。
3. **`AddSingleApeToScore` 门控过时**：原门控 `(!isCoreRowFirst || !isCoreLoopFirst)` 区分的是"左半来自 cacheTc（已含 ape）还是 workspace（裸）"；split 左半一律读裸 mm GM → 门控导致块首行的 score 左半缺 ape。修复：去掉该门控（保留 `sIdx!=0 && compressTcSize>0`）。
4. **跨 pipe race（偶发）**：split 引入 V-pipe 生产者（Cast bf16→fp32）写入被 copy pipe 消费的 buffer，原算子生产者全是 MTE2 同 pipe 有序从未暴露。表现为随机行/state 块损坏、加打印即消失。

   最终同步方案（与原始纪律对齐：队列自动同步 + 定向 flag，仅两个实证必需的全排空点）：
   - `FromWokrSpaceToUb`：stage copy(MTE2)→Cast(V) 之间一对 `MTE2_V` flag（裸 copy/向量混用）；**Cast 后一个 `PipeBarrier<PIPE_ALL>`**（实证必需：Cast 整块读写队列 buffer，队列 free 事件不保证其读排空，去掉后随机行损坏）
   - `LoadFromWorkSpace`：GM copy→Cast 之间一对 `MTE2_V` flag
   - `OverLap`：**SaveState 前一个 `PipeBarrier<PIPE_ALL>`**（实证必需：去掉后最后一个 slice 窗口确定性损坏，state 却正常）
   - 曾被怀疑后经二分证伪的屏障（pre-score 复用、pre-kv 复用、LoadFromWS 尾部 V_MTE2）均已去掉
   验证：decode_B8 12/12、2step state 6/6、全量套件多轮全绿。

另修复测试假象：输出行数 = min(T, T/r+B) 多于实际写入行数（未写行是未初始化内存），`compare` 改为只比写入行；2step 用例组未完成一行不写（只比 state）。

**验证结论**：split 输出 vs torch 参考（bf16 mm 输入）= **0.0000% rel>1%**（prefill/decode 均验证）；与 fused 的残差 ~11.5% rel>1% 全部是 MatMulV3 输出 bf16 vs fused cube 内 fp32 workspace 的固有精度差（实测上限 10.8-11.4%）。多轮重复跑无 race。

<details><summary>原 bug 记录（已解决）</summary>


**现象**（`test_compressor_split.py`，fused vs split，state0=0）：

| 用例 | 结果 |
|------|------|
| prefill B=1 q=8192 | 66% 行 mismatch，有 nan |
| prefill B=4 q=512 | 66% mismatch |
| decode B=8 | 99.9% mismatch（全错） |
| **decode B=64** | **0.72% mismatch（基本正确，bf16 舍入量级）** |
| chunk B=2 q=300 | 71% mismatch |
| decode 2-step B=8 | 86% mismatch，有 nan |

**最小复现**：q=4 单压缩 token（无 overlap 加载、无 LoadFromWorkSpace）就错（maxd≈8）。

**已排除**：
- mm 加载路径（FromWokrSpaceToUb + Cast）：`SaveState` 写出的 state_cache 与 fused 一致（max diff 0.03，bf16 舍入级）→ scoreUb/kvUb 内容**正确**
- vec 块其余代码、driver、tiling 与原算子逐字节一致
- usedCoreNum 语义（=20，vec 内部 ×2）已修正，不是本 bug
- Cast 暂存重叠：已证明写字节 4i < 读字节 2N+2i（i<N）安全，且 state 匹配再次佐证
- 探针（one-hot x + 随机 wkv）：split 输出**不等于任何行子集的均值**（fused = 4 行 coff1 均值，0.0003 对上）；split 的 nope 区也不是 fused 的通道移位/块置换 → 窗口里混进了垃圾行

**重点怀疑**（按优先级）：
1. **PadAlign 窗口装配的输入前提**：它假设 scoreUb/kvUb 的行是「段内连续」的；检查 `statisticInfo.dealSeqCnt`（我的 FromWokrSpaceToUb 加载行数）与 overLap 迭代器各 slice 的行偏移在 split 下是否一致（尤其多 slice/多核段时）
2. **DuplicateFirstBlock / ReadState 的 isFirst 分支**：q=4 时唯一会跑的特殊路径，若 split 下 slice 状态（isFirst/headHolderSeqCnt/compressTcSize）与 fused 不同 → 窗口 padding 行变垃圾
3. **vec1Res 行放置**：`CopyOutVec1Res` 的 `compressedCnt_` 计数在 split 的多核分布下是否与 vec2 的读取一致（B=64 偶然全对说明跟 dealTcNum 分布强相关）
4. decode_2step 的 nan → state 连续性问题（chunk/decode 续算路径）

**调试手段限制**：不允许在 kernel 里加 debug dump 写 GM。可用纯 python 探针（见 `dbg6_compressor_window_probe.py`：one-hot x + 随机 wkv + wgate=0 + ape=0 → 输出应是行子集均值，可反推窗口内容）。

</details>

## 5. 编译与测试

### 编译安装（唯一推荐，详见 AGENTS.md「编译与安装」）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=910b MAX_JOBS=256 pip install -e . --no-build-isolation --no-deps   # --no-deps 必须（离线）
```

增量（只改 compressor_epilogue 时，~5 分钟）：

```bash
cd csrc
# kernel 源码会被拷贝到 build/binary/.../src/<op>/ 并打 .done 标记，改了源码必须删拷贝
rm -rf build/binary/ascend910b/src/compressor_epilogue \
       build/binary/ascend910b/gen/compressor_epilogue_ascend910b*.done \
       build/binary/ascend910b/gen/kernel_meta_CompressorEpilogue_*
bash build.sh --pkg --ops="$(paste -sd';' /tmp/ops_910b.txt)" --soc="ascend910b" -j256
# 安装（注意 --install-path 必须绝对路径；会清空 vendor 目录，所以必须全量 ops 列表）
./build/cann-ops-transformer-custom_linux-aarch64.run \
  --install-path=/home/x50061890/vllm-ascend/vllm_ascend/_cann_ops_custom
```

`/tmp/ops_910b.txt` = build_aclnn.sh 里 910b 分支的 35 个算子列表（若丢失从 `csrc/build_aclnn.sh` 的 `CUSTOM_OPS_ARRAY` 重新提取）。

### 测试（卡 7，卡 0 有常驻负载、卡 1 内存不足）

```bash
cd /home/x50061890/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=7 python test_compressor_split.py     # 数值对照（fused vs split，6 场景）
ASCEND_RT_VISIBLE_DEVICES=7 python dbg6_compressor_window_probe.py  # 窗口内容探针
```

### 性能验证（已完成，2026-08-06，卡 1 空闲，msprof task_time）

- **prefill M=8192**：fused 2001.7us → split **1592.1us（1.26×）** = MatMulV3 559.6us×2（87.5% SOL）+ CompressorEpilogue 472.9us
- **decode**：epilogue 22.6us vs fused 47.6us（2.1×，另有微小 GEMM）
- 余量：epilogue 473us 高于 100-200us 预估——`PipeBarrier<PIPE_ALL>` 按正确性优先加粗，后续可按 pipe 对窄化（MTE2_V / V_MTE2 flag）收性能；GEMM 87.5% SOL（N=1024 偏小），可考虑 wkv|wgate 合并单 GEMM [M,7168]×[7168,2048]

### 性能验证 v2（已完成，2026-08-06，完全串行化重构）

epilogue 从**两阶段（vec1+SyncAll+vec2）重构为单阶段完全串行**：每核独占完整 headDim（行并行），rms_norm/rope 移入核内，删掉全部 SyncAll / vec1Res workspace / vec2 阶段 / 双缓冲簿记（详见 `csrc/attention/compressor_epilogue/README.md`）：

| 版本 | prefill M=8192 epilogue | prefill 总 | wait_id14（事件等待） |
|---|---|---|---|
| fused 单算子 | — | 1983.3us | — |
| v1（16×SyncAll） | 454.98us | 1568.7us | 103.4us |
| **v2（完全串行）** | **248.74us** | **1380.1us** | **5.7us（-94%）** |

- fused → v2 = **1.437×**；v1 → v2 = 1.137×；vec 占用 21.5% → 38.3%（计算量不变，等待消失）
- 剩余瓶颈：GEMM 2×564.6us（86.7% SOL，主导）+ epilogue 248.7us（内存下限 ~66us，余 ~58us 为 2 处 PIPE_ALL + flag 指令等待 + icache 4.1%）
- 数值回归：`test_compressor_split.py` 全绿（prefill rel>1% ≈ 11.6% = bf16 mm 固有精度差，decode ≈ 0.6-0.9%，state 全对齐，2step 6/6）

```bash
cd benchmarks/ops_profiling
VLLM_ASCEND_DSA_COMPRESSOR_SPLIT=1 ASCEND_RT_VISIBLE_DEVICES=1 timeout 900 msprof op --kernel-name="CompressorEpilogue*" --application="python bench_deepseek_v4.py --case dsa_compressor --iters 1 --warmup 0 --M-prefill 8192" --output=msprof_out/serial_op
# PipeUtilization：aiv_vec_time ≈ 91us（38.3%），aiv_mte2 ≈ 45us，aiv_mte3 ≈ 43us，scalar_wait_ib ≈ 89us，wait_id14 ≈ 5.7us
```


```bash
cd benchmarks/ops_profiling
VLLM_ASCEND_DSA_COMPRESSOR_SPLIT=1 ASCEND_RT_VISIBLE_DEVICES=7 bash run_msprof.sh bench_deepseek_v4.py
# op_summary 里 MatMulV3 ×2 + CompressorEpilogue 三行合计 vs 融合 Compressor 的 1926us (prefill M=8192)
```

## 6. 涉及文件清单

- 新增：`csrc/attention/compressor_epilogue/`（op_kernel + op_host + CMakeLists）
- 修改：`csrc/torch_binding.cpp`、`csrc/torch_binding_meta.cpp`、`csrc/build_aclnn.sh`（CUSTOM_OPS_ARRAY）、`csrc/cmake/third_party/ascend_protobuf.cmake`（--parallel）、`csrc/attention/compressor/op_kernel/arch32/compressor_template_tiling_key.h`（SEL 裁剪）、`csrc/attention/compressor/op_host/CMakeLists.txt`（DAY0_SCOPE）、`vllm_ascend/envs.py`、`vllm_ascend/attention/dsa_v1.py`、`benchmarks/ops_profiling/bench_deepseek_v4.py`
- 测试：`test_compressor_split.py`（已修：只比写入行 + main 守卫）、`dbg6_compressor_window_probe.py`（仓库根目录，事后清理）
