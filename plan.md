# DeepSeek V4.1 Flash：910B × 8 适配计划

调研日期：2026-09-15。目标仓库：个人 fork `https://github.com/Qeeweew/vllm-ascend.git`。
开发分支：`deepseek-v41-910b-w4a16-engram`，已从本地 `main` 创建。

本文包含实施计划和验收记录。截至2026-09-16，48/48权重转换已完成；完整40层TP8 eager/graph、366.22GiB真实pinned Engram及五项文本graph smoke（含4243-token检索）已通过，详见 `benchmarks/deepseek_v41/FULL_MODEL_RESULT.md`。

融合indexer已从r14的B1特化重写为按query调度的r17，真实ragged多请求、16次输入变化graph replay、T1024 prefill通过。性能尚未通过：独占卡T128/32K约1.33ms，旧QLI约1.07ms；T512/4K退化2.76×，而T128/128K与T512/128K分别加速约1.64×/1.58×。旧QLI的M128在4个query间复用K，新candidate M32失去该复用，并有全query预处理等待；正在对照原 `qli_opt` 与当前QLI源码及全核profiling，再决定tiling/流水修改，不把长上下文收益外推到全部prefill。

Router r3的84项NPU/graph正确性和48组独占卡性能测试通过；RoPE/cache r12的193项NPU、33项CPU和64组独占graph性能测试全部通过，plain RoPE T1024/H32/D128由243.158降至40.879µs。融合算子注册、Meta与模型接线已提交`00ce95d69`，189项框架CPU检查通过，完整模型融合验证待执行；生产安装仍是r12。DSpark专用AscendC metadata已通过58项NPU测试和96次变输入graph replay；真实TP8 proposer r9的15组场景（含context33、255/256及拒绝0–5）在8rank全部通过，CPU Markov选词75token完全一致，全部worker正常退出。原AICPU根因仍未确认，替换路径已解除该集成阻塞；当前proposer回归为B1/eager，完整DSpark context/query graph、多请求proposer和target/scheduler联动尚未完成。AICPU增量构建漏重链接问题已修复并提交`815a8bf02`。DSpark必须开启且draft计算需要graph，具体context/query独立捕获边界与验收见[DSpark graph计划](benchmarks/deepseek_v41/DSPARK_GRAPH_PLAN.md)。完整模型视觉、native W4A16整模、DSpark、长上下文及最终 `vllm bench` / 整机profiling尚未完成。后续早期状态保留为调研记录，不能视为最新进度。后台编译与外围接入并行推进。

## 1. 目标与总体决策

1. 推理代码、转换工具、AscendC 算子、测试均放在 `vllm-ascend`；`sources/vllm`、`sources/ops-transformer`、`sources/catlass` 用作参考。
2. 910B3 × 8，每卡 64 GiB。首个运行基线采用 TP8、PP1、DP1，保留 TP8 + EP 的对照路线。初始上下文 8K，逐步验收 32K、128K，再评估 1M。
3. 普通 FP8 权重正确反量化到 BF16；已有 FP32 控制参数保留 FP32。主干 routed MoE 使用 INT4 group32 signed-scale RTN，激活 BF16；shared expert 首版使用 BF16。
4. Engram 大表常驻 host，首版以 BF16 pinned host 分片 + 小型 pinned staging + 固定地址 NPU staging 实现。查表和传输在 graph 外，模型 graph 内只读取已准备好的 BF16 rows。
5. 先完成文本、eager、标准自回归正确性，再完成 NPU graph、性能优化、视觉、DSpark 和长上下文。后几项属于最终适配范围，不把首个文本基线视为整个任务完成。
6. 新增设备算子使用 AscendC。小批量 GEMV 与大批量 Cube grouped GEMM 分开优化；CATLASS 提供矩阵乘基础组件，不引入 CUDA/Triton 新算子作为交付实现。
7. **用户指定：compressor 算子重写，旧实现不作为正确性基础；矩阵乘法必须独立拿出来。** 保持模块职责简单，先做独立 projection、CR1/CR2 压缩/归一化、cache insertion，不延续旧算子复杂流水。
8. **用户授权使用独立 sub-agent 开发算子，并要求严格性能验收。** 主任务负责数值契约、接口、集成与最终验收；算子任务在明确文件范围内交给sub-agent。全部功能适配完成后继续做整机profiling和优化，形成可复现报告，这是交付的一部分。
9. **DSpark必须开启，不能以普通自回归回退作为交付。Draft计算进入NPU graph并验证实际replay也是最终验收目标，不能仅交付eager实现。** 必须通过真实三层draft proposer、target verification、接受/拒绝及回滚、target/draft NPU graph协调和服务端请求，并单独报告DSpark端到端性能。组件oracle和标准自回归通过不能替代这些验收；当前metadata故障必须修复后才能开放生产准入。
10. **融合indexer以多batch为主要验收范围。** B1局部优化和旧路径B8/B32回归不能作为交付。重新设计请求与候选块的核分配，至少覆盖B1/2/4/8/16/32、H32、CR1/CR2及混合请求长度；必须证明各形状实际进入融合路径，独立报告正确性、graph、完整selector延迟、吞吐、HBM和Cube/MTE/核负载指标。
11. **按用户最新澄清，top-k仍融合，解耦Cube与Vector任务。** Candidate consumer的AIC分块完成分页INT8读取、QK、缩放/ReLU及head加权归约；单个AIV核负责一个query完整16384项分数的top-k及逻辑位置映射。用明确的分数缓冲、完成信号和生命周期协议，使AIC可继续计算后续query，AIV消费已完成query；重新计算排序scratch及192KiB UB预算。此前外部 `torch.topk` 提案已被此要求替代。禁止重新物化全量BF16候选key或多头QK，禁止用旧路径多batch回归代替新路径验收。16384上限仅适用于candidate consumer，source与CR2按各自实际语义处理。
12. **融合indexer以prefill性能为首要优化目标。** 按有效query token数而非仅batch数选择tiling：token充足时优先不切分16384候选，依靠query间并行；token较少时比较有限split数，阈值和最大split数须由实测决定。覆盖单请求长prefill、多请求不等长prefill、混合prefill/decode和decode，验证每query causal可见范围。报告完整selector的prefill tokens/s、median/P95、Cube/MTE/Vector/同步等待和各核负载，不以decode局部收益代替prefill验收。

## 2. 已核对的环境与源码基线

| 项目 | 实际状态 |
| --- | --- |
| 个人 vllm-ascend 基线 | `b49962987e89b850586f1819ce8f85daa85a0f81` |
| 官方 vLLM 参考 checkout | `836bb3839ffefcda8283ea7d41671a89e1a613df`，2026-09-15，含 V4.1 CUDA 实现 |
| 已安装 vLLM | `0.27.1+empty`，editable 指向 `/vllm-workspace/vllm`，源码 HEAD `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`，未发现对应 V4.1 模块 |
| 已安装 Ascend 插件 | editable 指向 `/vllm-workspace/vllm-ascend`，该目录有其他开发工作，不能覆盖 |
| 当前 fork 的 vLLM release 标记 | `.github/vllm-release-tag.commit` 为 `v0.28.0` |
| 当前 fork 的 verified main 标记 | `84030bbe3d74d99bad477a3d2e37a973ccd8865c`，本地官方浅克隆尚无此对象；不能当作已经验证当前参考 HEAD |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4`；此 CPU wheel + torch-npu 组合可以使用 NPU |
| Transformers | `5.14.1` |
| CANN / driver | CANN `9.1.0`，driver `25.5.0`，aarch64 |
| 硬件 | 8 × Ascend 910B3，各 65536 MiB HBM；调研时每卡已有约 3.4 GiB 占用 |
| Host 内存 | 总计约 2 TiB，available 约 1.9 TiB |
| 文件系统空间 | `/mnt/models` 可用约 1.4 TiB，工作区所在文件系统约 489 GiB；转换输出放模型盘 |
| memlock | `ulimit -l` 为 65536 KiB；小块 pin 已成功，但大规模 pin 能力需要单独测量 |
| CATLASS 参考 | `1e5684737b8aa544534b735b3e179ff8ec21bfca` |
| ops-transformer 参考 | `a041b05638a5a3f94b79c97ce196a331738a56a1` |

构建依赖必须使用 fork `.gitmodules` 指定的 CATLASS `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，不能将参考 checkout 的不同 HEAD 静默替换进去。

### 2.1 环境处理

已创建工作区 `.venv`，继承容器 site-packages，以复用匹配 CANN 的 torch/torch-npu，同时隔离 editable 安装。系统环境仍服务原有工作目录。

已成功执行的完整构建命令（工作区根目录）：

```bash
SOC_VERSION=ascend910b3 MAX_JOBS=32 \
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
.venv/bin/python -m pip install -e ./vllm-ascend --no-deps --no-build-isolation
```

日志：`/tmp/deepseek-v41-ascend-install.log`。走 `setup.py → csrc/build_aclnn.sh → CMake / torch extension` 全量流程，不能把跳过 custom kernel 的安装当作算子构建完成。

参考vLLM也已用以下命令安装成功；先补齐 `setuptools-rust`，并将setuptools固定到满足上游约束的80.10.2：

```bash
VLLM_TARGET_DEVICE=empty .venv/bin/uv pip install \
  --python .venv/bin/python --no-deps --no-build-isolation -e ./sources/vllm
```

使用开发环境：在工作区执行 `source .venv/bin/activate`。参考vLLM浅克隆缺少版本标签，当前生成版本为 `0.1.dev1+g836bb3839.empty`，插件为 `0.1.dev5255+gb49962987`；这是SCM元数据结果，不代表实际源码能力。进入模型运行阶段前必须补齐可信tag/版本来源并检查版本分支判断，不伪造版本号以跳过兼容性检查。

依赖审计已经发现明确冲突：当前 fork 要求 `fastapi<0.124.0`，官方参考源码要求 `fastapi>=0.133.0,<0.137.0`、`starlette>=1.0.1`；原容器实际为 FastAPI 0.123.10 / Starlette 0.50.0。官方参考 `pyproject.toml` 的构建依赖还指定torch2.13，而NPU栈是torch2.10/torch-npu2.10；因此保留已匹配NPU栈，采用empty/no-build-isolation进行兼容性检查，不能盲目升级torch。另有 OpenCV 5 与 NumPy 1.26、profiler 工具链依赖问题。`--no-deps` 只防止构建时重装 NPU 栈，不表示这些冲突已经解决。

后续按顺序完成：

1. 固定 vLLM + plugin 源码 SHA，检查 import/API 契约；参考 HEAD 含 V4.1，不能用旧安装的版本字符串代替源码能力检查。
2. 使用 `.venv/bin/uv` 管理参考 vLLM 的开发安装，`VLLM_TARGET_DEVICE=empty`，避免安装 CUDA 后端；不修改官方参考仓库业务代码。
3. 在个人 fork 调整经实际 API 检查证明可兼容的依赖约束；HTTP serving 的 FastAPI/Starlette 对齐单独验收。不能只强行安装冲突版本后宣布环境可用。
4. 检查 `vllm.__file__`、`vllm_ascend.__file__`、distribution editable 元数据、自定义 `.so` 路径、NPU op 注册、worker 子进程 import 和 `pip check`。
5. 区分继承自容器的 profiler 冲突与本任务新增冲突，记录实际结果。先验收 offline inference，再验收 OpenAI-compatible serving。

## 3. 模型配置与 V4 → V4.1 差异

真源：`/mnt/models/DeepSeek-V4.1-Flash/config.json`、该目录 `inference/model.py` / `engram.py` / `convert.py`，以及官方 vLLM `vllm/models/deepseek_v41/`。

关键配置：hidden=5120，40 主干层，64 attention heads，head_dim=512，RoPE dim=64，SWA=128；384 routed experts，top6，intermediate=2304，1 shared expert；`sqrtsoftplus` routing，scale=1.5，SwiGLU limit=10；mHC streams=4，Sinkhorn=20，RMSNorm eps=1e-20。主配置为 `DeepseekV41ForCausalLM`，内部有 text/vision 两层配置，不是旧 V4 配置改模型名。

| 模块 | 官方 V4.1 行为 | 当前 vllm-ascend 状态与工作 |
| --- | --- | --- |
| 模型入口 | `deepseek_v41/__init__.py` 目前按 ROCm / NVIDIA 分派；普通 else 会进入 NVIDIA | 尚无 V4.1 注册。必须提前注册 NPU 入口并隔离 CUDA 导入；不可只把 architecture alias 到 V4 |
| CSA2 | CR0/1/2；Full / Reindex / Reuse；main KV 与 top-k 跨层复用 | V4 indexer 的路径仍有 CR4/128 限制；需要改 metadata、cache owner、生产消费关系 |
| 两级 indexer | 第 20 层输出 2048 个候选块，每块 8 positions；后续 index source 在候选集内取 top512 | `quant_lightning_indexer_v2` 已有 candidate 输入/输出、mode 1/2/3 和 910B tiling 支持，应接入并验证，而非从零重写 |
| KV 共享 | KV source 为 2/8/14/20；index source 为 2/8/14/20/24/28/32/36 | `sparse_attn_sharedkv` 与 metadata 已有基础；仍需验证 V4.1 的跨层所有权和生命周期，不能按名字认定完成 |
| Compressor | CR2 有跨 chunk 的未闭合组状态；CR1 每 token 生成；latent 复用于 main KV 与 indexer K | 按用户要求重写 AscendC；矩阵乘法独立，压缩/归一化只做简单向量与状态处理，旧算子不作为正确性基础 |
| mHC | Single-Pass / delayed pre-mix，上一子层的 mix 供下一子层用；末层无 learned hc_head | 当前 `hc_pre` / `hc_post` 可作为基础，不能沿用 V4 forward 顺序 |
| Engram | 第 1、14 层；token n-gram 查表，随后 key/value projection 与 residual-dependent gate | 只有独立 `ops/triton/engram_int8.py` 辅助，不是完整 V4.1 host offload；需要服务状态、host storage、图边界与 AscendC gate |
| MoE | 384/top6、signed-scale INT4 目标、TP8 局部 intermediate=288 | 已有 W4A16 scheme；个人优化分支需要迁移并补维度/精度覆盖 |
| Routing | sqrtsoftplus、校正 bias、归一化及 routed scale；视觉存在不同 routing 信息 | 现有 router 已支持 sqrtsoftplus。复用后逐项核对，不重新实现已有能力 |
| Vision | 32 层、hidden1024、patch14、downsample3；image spans 影响 Engram 和 routing | 复用已有 V4 vision/preprocess 组件，验证 V4.1 checkpoint 命名、mask、aligner、position |
| DSpark | 3 层、block5、target37/38/39、128 experts/top3、Markov rank256、confidence head | 现有 V4 DSpark 不是可直接复用的完整语义；标准 decode 稳定后独立适配 |
| 服务协议 | V4.1 tokenizer、encoding、reasoning/tool parser | 官方仓已有 V4.1 tokenizer/parser；不能继续套用 V4 特殊 token 配置 |
| CED / bounded replay | 报告描述 encoder-decoder 结构与 prefill 优化、SWA bounded replay | 必须区分报告能力、参考 inference 实际行为、vLLM 已实现行为；首版完整 forward 保语义，优化另设验收 |

### 3.1 CSA2 层关系表（0-based）

| 层 | CR | main KV source | index source | 行为 |
| --- | --- | --- | --- | --- |
| 0–1 | 0 | 无 compressed KV | 无 | SWA |
| 2–7 | 2 | 2 | 2 | 2 为 Full，其余 Reuse |
| 8–13 | 2 | 8 | 8 | 8 为 Full，其余 Reuse |
| 14–19 | 2 | 14 | 14 | 14 为 Full，其余 Reuse |
| 20–23 | 1 | 20 | 20 | 20 为 Full，发布候选池，其余 Reuse |
| 24–27 | 1 | 20 | 24 | 24 为 Reindex，其余 Reuse |
| 28–31 | 1 | 20 | 28 | 28 为 Reindex，其余 Reuse |
| 32–35 | 1 | 20 | 32 | 32 为 Reindex，其余 Reuse |
| 36–39 | 1 | 20 | 36 | 36 为 Reindex，其余 Reuse |
| MTP 40–42 | 0 | 独立 draft SWA | 无 | DSpark，不能套用主干 cache |

源层持有 cache，消费者使用显式引用；每次 forward 的 top-k/candidate buffer 不能误当持久 KV。请求重排、prefix reuse、抢占恢复和 graph padding 都必须更新 metadata，避免复用上一请求的索引。

## 4. 权重盘点与容量预算

已只读扫描 safetensors 头部与 `model.safetensors.index.json`，没有载入整模型。索引 `total_size=510286023000` bytes。48 个分片中，前 46 个文件的实际大小与 header 推导大小一致；47/48 尚不存在，它们分别是 layer1/layer14 Engram 表。文件长度检查不等价于内容 checksum 校验，下载完成后还需最终清单校验。

### 4.1 实际 checkpoint 类型

| 类型 | 已读出的实例 | 转换策略 |
| --- | --- | --- |
| Routed expert FP4 | `layers.0.ffn.experts.0.w1.weight` 为 I8 `[2304,2560]`；scale 为 F8_E8M0 `[2304,160]` | I8 是打包 E2M1，并非 signed INT4；解码后再 RTN |
| 普通 FP8 | `layers.0.attn.wkv.weight` 为 F8_E4M3 `[512,5120]`；scale `[16,160]` | 32×32 block 反量化到 BF16 |
| Shared expert | F8_E4M3，32×32 block scale | BF16，首版不额外量化 |
| 控制/归一化参数 | 存在 BF16 和 F32，含 attn sink、mHC 等 | 按模型数值契约保留；不能全局 `.bfloat16()` |
| Engram table | 文件未到；config 与参考代码指向 FP8 rows + 每行 group32 scale | 收到 header 后确认形状/dtype；预计 BF16 host 总计 366.22 GiB |
| MTP | 独立 `mtp.*` namespace，含 I8 routed expert、F8 scale、BF16/F32 参数 | 单独 manifest；主干基线可不加载，DSpark 阶段启用 |

实际 key 是 `layers.*.ffn.*` / `.attn.*` / `.scale`，不应假设原始 checkpoint 已是 HF `model.layers.*.mlp.gate_proj.weight_scale` 格式。

### 4.2 内存与磁盘

主干 routed MoE 参数量为 `40 × 384 × 3 × 5120 × 2304`。INT4 数据 253.125 GiB，BF16 group32 scale 31.640625 GiB，总计 **284.765625 GiB**，理想均分 **35.5957 GiB/rank**。这是权重存储下界，不含临时 repack、offset、padding 和通信。

Engram 参数量为 `(384006168 + 384016682) × 256 = 196613849600`，BF16 为 **366.2218 GiB**，按 TP8 hash-head 分片约 **45.7777 GiB/rank host**。预期原始 FP8+UE8M0 为 188.8331 GiB，须待分片核实。禁止八个 rank 各加载一份完整 Engram 再切片，也禁止将整个 Engram 临时 `.to(npu)`。

首版设备预算以每卡约 36 GiB routed MoE 为起点，另外统计 dense/shared BF16、mHC、vision、KV、HCCL、graph pools、staging、workspace。实际每卡应保留至少约 6–8 GiB 余量再增大 batch/context；不能只用总 HBM 减权重推算最大并发。

BF16 main compressed KV 在不分片时，每序列每 rank 约为 `1024 bytes × (3/2 + 1) × L`：32K 约 80 MiB，128K 约 320 MiB，1M 约 2.5 GiB。还必须加 indexer K、40 层 SWA、CR2 ring、页对齐、allocator 和并发；该公式不是总 KV 预算。报告中的 890 bytes/token 对应特定量化格式，不能照搬到 BF16 基线。

全套转换预计约 650–700 GiB（BF16 Engram + INT4 MoE + BF16 其他权重及可选 MTP），最终以 manifest 实算。保留原始约 475 GiB checkpoint；输出放 `/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32`，分片写入、原子 rename、断点续转，不能生成完整 BF16 MoE 中间副本。

## 5. 实施顺序与验收门槛

| 阶段 | 工作与产物 | 完成条件 |
| --- | --- | --- |
| P0：基线固定 | 分支、环境清单、参考 SHA、完整构建、依赖审计、本计划 | 来源可复现，区分 import 成功与完整构建成功 |
| P1：格式与原型 | 转换器、量化数值契约、Engram host/staging 原型、真实 288 维 W4A16 验证 | pack/unpack 无损；signed scale 保真；动态 rows graph replay 正确 |
| P2：模型 eager | V4.1 入口、mHC、CSA2、Engram、loader、TP8 MoE | 文本 8K、40 层加载完整，逐层和 logits 对齐，无关键权重漏载 |
| P3：graph | 图外 prefetch hook、固定 buffer、padding/mask、full decode graph；piecewise 性能路线 | 变化输入、batch 重排、回收与并发下不读旧 rows；eager/graph 一致 |
| P4：性能 | 288 尾块 AscendC、小 batch kernel、Cube GMM、NUMA/拷贝重叠、CSA2 算子优化 | 给出 TTFT/TPOT/吞吐、误差与峰值内存；每项收益有 profile 支持 |
| P5：能力补齐 | Vision、DSpark、prefix/chunked prefill、长上下文、CED/SWA replay 优化评估 | 逐项 E2E 和服务测试通过，明确实测上下文与功能组合 |
| P6：整机profiling与交付 | 全部适配后的profiling、瓶颈优化与复测；完整build/install、测试/量化质量/性能报告、启动脚本、个人fork变更集 | 真实负载达到冻结性能门槛，优化无精度回退，从干净环境可复现 |

下载未完成时可以推进 P0/P1、合成小模型、层级 reference 和算子工作；最终 Engram 大表验证及全模型 P2 依赖完整下载。每完成一个阶段更新本文件的实际状态，不用估计时间代替验收。

## 6. 权重转换的数值与存储契约

算法依据：工作区 `docs/MXFP4-aware Signed-Scale INT4 RTN.md`。以下显式写出约定，避免 converter、loader 和 kernel 各自解释 INT4。

### 6.1 FP8 → BF16

对于普通二维矩阵，先读取对应 `.scale`，根据实际 weight/scale shape 确定 block。在当前已读权重中为 32×32：

```text
W_bf16[o,i] = BF16(FP32(W_fp8[o,i]) * FP32(S[o//32,i//32]))
```

UE8M0 必须按浮点 scale 解码，不能把存储 byte 的整数值直接当 scale，更不能因为某些 loader 将参数称作 `weight_scale_inv` 就误做除法。特殊 exponent/非有限值按实际 format 检查并报告；不能悄悄 clamp。首选 CPU torch 对 E8M0 dtype 的正确转换，另用小型独立解码 reference 验证。

Engram 是 row-wise group32：`S[row, channel//32]`，与 dense 的 32×32 block 不同。`wo_a` 的 grouped 输出布局、vision、aligner、shared expert 和 Engram projection 分别处理，不用一个全局 reshape。已有 F32 mHC、router、sink 等参数按 forward 所需保留。

### 6.2 MXFP4 → signed-scale RTN

E2M1 nibble codebook 按参考 `inference/convert.py`：

```text
code 0..15 = [0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]
byte 的低 nibble 是偶数 K 元素，高 nibble 是后一个 K 元素。
W[o,k] = BF16(codebook[code[o,k]] * UE8M0_scale[o,k//32])
```

每个输出 row 沿逻辑 K 轴每 32 个元素分一组。必须在任何 transpose、TP slicing 或 NPU layout repack 前明确这个 group 轴。先得到 BF16 reference，再用 FP32 做 min/max、scale 求解和 rounding：

```text
a = -min(min(W), 0)
b =  max(max(W), 0)
s =  a/8       if a > b
    -b/8       if b > a
    -b/7       if a == b and b != 0
q = clamp(round_to_nearest_even(W / s_stored), -8, 7)
W_hat = FP32(q) * FP32(s_stored)
```

`s_stored` 为 BF16 signed scale；先确定其最终存储值，再量化 q，保证离线 reference 与 kernel 使用相同 scale。全零组令 q 全零、scale 取有限非零值；首版固定用 FP32 epsilon 再转 BF16，并写入格式版本。非零 scale 若转 BF16 后变零或非有限则报错。比较 `/7`、`/7.5` 和 signed-scale 时统一上述存储 rounding 约定。

负 scale 不能取绝对值。`q=-8` 与负 scale 对应正侧最大值，直接把 q 全部取反会遇到 +8 不可表示。物理 INT4 仍是 `[-8,7]`，zero point 为零，不是 AWQ 的 affine zero point。

真实抽样（layer0/expert0，全矩阵，BF16 scale，NRMSE=`||W_hat-W||2/||W||2`）：

| 矩阵 | Signed-scale | `/7`，q∈[-7,7] | `/7.5`，q∈[-8,7] |
| --- | --- | --- | --- |
| w1 | 0.068609 | 0.100219 | 0.084983 |
| w2 | 0.067857 | 0.099741 | 0.084129 |
| w3 | 0.068541 | 0.100151 | 0.084907 |

该抽样约 68% groups 为负 scale、36% 为 amax tie，说明负 scale 与 tie 分支不是罕见边界。它只证明这三个权重样本的误差改善，不保证所有层、激活误差、困惑度或任务质量必然改善。

### 6.3 必须区分两种 INT4 packing

当前 `quantization/methods/wna16/w4a16.py::unpack_from_int32` 解码 nibble 后会减 8。因此其 compressed-tensors checkpoint 路径使用 **offset-binary `q+8`**；个人优化 kernel 的原生 layout 则使用 **two's-complement `q & 0xF`**。两者不能直接混用。

首版沿用现有 compressed-tensors loader 契约：

| 阶段 | weight | scale |
| --- | --- | --- |
| 单专家 canonical checkpoint | `[N,K/8]` INT32，K 轴打包 8 个 `q+8` nibble | `[N,K/32]` BF16 signed |
| 合并 gate/up 后 | 明确 `w1=gate`、`w3=up`，按 loader 约定堆叠 | 同序合并 |
| 原生小 batch kernel | `[E,K,N/8]` INT32，N 轴 two's-complement packing | `[E,K/32,N]`，支持负值 |
| CANN GMM | 按当前 npu int4pack/NZ 接口要求转换 | 与 K 分组和 N 布局一致 |

转换器输出单独的格式 manifest，至少记录 `format_version`、source SHA/config hash、qrange、group axis/size、signed-scale、rounding、scale dtype、packing encoding、矩阵逻辑 shape、tensor key 映射、输出 checksum。若 compressed-tensors schema 对负 scale 有额外验证限制，应在 Ascend 自有 metadata/loader 中显式表达并验证，不能声称任意外部 compressed-tensors consumer 都兼容。

模型 config 要切换到真实 INT4 MoE / BF16 dense 的量化配置，只给 routed experts 配 W4A16，其他层明确忽略。去掉会重新触发 `deepseek_v4_fp8` / MXFP4 dispatch 的原始全局量化声明。HF top-level/text_config 的相关字段同步处理，避免一处残留导致错误创建 FP8 参数。

### 6.4 转换与加载的资源控制

1. 输入只读；按 index 追踪 weight 与 scale，即使它们分属不同 shard。先验 shape、dtype、文件完整性再读取。
2. tensor 级、row-block 级转换。禁止整层 384 个专家先展开 INT32，更不能八个 rank 同时展开整个模型。现有 post-load 大张量 unpack/repack 要改成分专家或小批量转换。
3. 直接输出目标 INT4 + scale；BF16 MoE 只存在小块临时 reference，不生成第二套数百 GiB dense MoE。
4. 转换分片写 `.partial`，完成校验后原子 rename，最后发布完整 index/manifest。断点续转以 source/参数 fingerprint 与 checksum 为准，不能只检查同名文件存在。
5. Engram 按 hash-head 的 prime bucket 范围做 TP8 分片，不能按均匀 row slice 后假设等价；不同 head 的 bucket 长度略有不同。大表直接转换到指定 host backing file，启动时只映射本 rank 的 head ranges。
6. 原始 `.scale` 只有在其 weight 确认成功转换后才从输出集合删除；权重漏载/多余权重按显式 allowlist 报告。可暂不加载的 MTP/vision 参数必须列明，不能把任意 unexpected key 都忽略。

## 7. W4A16 MoE 分支迁移与 910B 算子路线

已定位 `origin/deepseek-v4-w4a16-tp`，核心提交：
`274dd30c3230804e35b7985e288dacbde3326ece`，`feat(moe): add fused W4A16 TP kernel`。

该提交已有 signed group32、BF16/FP16 激活、SwiGLU clamp、torch/meta 注册和小 batch 测试。分支同时夹带其他旧版本兼容变更，不能整分支 merge/cherry-pick 后覆盖当前主干。以此提交为参考迁移必要 h/cpp、binding、quant scheme 和 dispatch；旧 `quantization/methods/w4a16.py` 已迁移至 `methods/wna16/w4a16.py`。

### 7.1 必须先解决的具体问题

1. **288 维尾块**：TP8 将 intermediate 2304 切成 288，而旧 tiling 检查 `intermediate % 128 == 0`。新 kernel 必须支持 288 的真实 K/N 尾块，含 gated w13 的 576 维输出、w2 的 K=288 与 scale 的 9 个 groups；不能只删除 tiling assert。临时 padding 到 384 会让相关 MoE 存储/计算增加约 33%，仅作诊断对照，不作为默认优化。
2. **BF16 路径精度**：旧测试 reference 明确使用 FP16 group accumulator，允许约 8% 相对误差。检查 kernel 内部 BF16→FP16 转换、溢出、group partial sum 与最终 FP32 累积；用独立高精度 GEMM reference 衡量，不仅对齐复刻 kernel rounding 的 reference。
3. **路由权重只乘一次**：验证 input/output gating 位置、SwiGLU clamp（gate 上界10；up ∈[-10,10]）、routed scale1.5、shared expert 合并、TP reduce。shape 对齐不能替代这些语义检查。
4. **静态 graph workspace**：核对 Torch adapter 的临时分配、contiguous、workspace 生命周期、Meta 输出 shape；动态 batch/expert 分布变化不得触发未捕获的 host 分支。
5. **不要保留两份完整 packed weights**：small-batch 和 GMM layout 如果不同，应优先统一或只保留必要 layout。双份权重会破坏 64 GiB 预算。

### 7.2 两条计算路径

**小 batch decode**：复用个人分支 AscendC AIV GEMV 路径，只优化 decoding，修复 288 尾块与 BF16 数值问题，覆盖 B=1/2/4/8/16/32/64 后再用测量选阈值。路由 token 分布会改变收益，阈值不能简单沿用旧模型。此 direct-launch kernel 使用 `SyncAll`，必须保留原分支的 `KERNEL_TYPE_MIX_AIV_1_0` 以生成 FFTS 地址参数；仍只使用 AIV 计算。改成 `AIV_ONLY` 的首测出现同步挂起，恢复原模式后 B1/B2/B8 精度通过。

**Prefill / 大 batch**：按用户最新要求，保留现有 `torch_npu.npu_grouped_matmul`，不另写 prefill kernel。真实 H5120/I288、BF16 negative scale、q=-8、group32 已通过 CANN 数值检查。decode dispatch 必须显式确认阶段，不能只因 token 数量少就将小 prefill 送入新路径。

参考 `sources/catlass/examples/02_grouped_matmul_slice_m/`、`include/catlass/gemm/` 以及 ops-transformer `gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant.h`。只使用经核对适用于 arch22 的组件，不能把 `ascend950_*`/MXFP 原生算子当成 910B 能力。

若采用 CV 混合核：AIV 做 INT4 unpack/signed-scale 反量化，AIC 做 BF16 matmul，使用有界 GM workspace 双缓冲与明确生产消费事件。开发前按技能复读事件配对/跨核同步规则；先实现无重叠正确版本，再重叠 MTE/Vector/Cube。group tail、empty expert、重复 expert ID、padding token 都需有明确处理。

### 7.3 TP8 与 EP 对照

默认先打通 TP8，不启用额外 expert parallel：每 rank 全部 expert 的局部中间维288，通信为既有 TP 路线。第二条比较为 TP8 attention + EP8 routed experts：每 rank 48 个完整 intermediate2304 专家，减少中间维尾块压力，但需要正确的 token dispatch/combine、局部 expert ID 与 HCCL 通信实现。EP 能否高效、是否与当前图捕获路径兼容必须实测，不能把 EP 当作无成本避开288的办法。

首版不开启 EPLB、elastic EP 或多机。比较真实请求的 TTFT、TPOT、吞吐和通信比例后决定默认拓扑。

## 8. Engram host offload 与 NPU graph

### 8.1 为什么 CUDA 实现不能直接搬运

官方 `nvidia/engram.py` 使用 `cudaHostRegister`、UVA device view、CUDA stream，以及 `@eager_break_during_capture` 让 lookup 跨 piecewise graph segment 运行；`EngramConfig.verify_model_config` 还明确限制 CUDA。

pin_memory 只保证 host 缓冲可用于相应 DMA 路径，不证明 NPU AscendC kernel 可以解引用 CPU pointer。CANN 9.1 确实声明了 `aclrtHostRegister` / `aclrtHostRegisterV2` / `aclrtHostGetDevicePointer`，但 API 存在不代表 910B 上任意 mapped pointer + 随机 gather 可行或高效。UVA 式直接访问作为可选硬件实验，不能作为首版依赖。

也不能在模型 forward 中写 Python CPU gather，然后期待 replay 再运行一次：图 replay 通常不执行这些 Python 语句。仅 capture 成功不足以证明 offload 正确。

### 8.2 存储与分片

每层 `(4-1)×8=24` 个 hash heads，每个 token 查 24 行，每行256 BF16。TP8 每 rank 负责3个 heads，host 只持有这些 heads 的 prime bucket 区间。首版 DP1 无需 DP shared-memory 服务。

首选每 rank 约45.78 GiB BF16 host pinned 表（两层合计），按 NUMA 归属分段分配/注册，直接装入最终 host storage，避免先分配 pageable 大表再 `.pin_memory()` 产生双份峰值。逐级验证 pin 上限、进程退出回收、NUMA first-touch、连续运行的系统内存压力；当前64 MiB memlock不能单凭数值判断 allocator 行为，小块成功也不能外推整表成功。

若实际无法 pin 整表，显式选择 bounded-pinned 模式：BF16 mmap/pageable 大表 + 固定容量 pinned staging，CPU gather 后 DMA。它仍实现 host offload，但整表未锁页，首次访问/page fault 与 NUMA 性能不同，必须在配置/报告中说明，不静默降级。无需借助量化成 INT8 来规避 pinned 内存设计；当前 int8 helper 不作为默认精度路线。

每 rank 维护两层的 host staging、固定地址 device rows、valid token mask 和事件，容量由预先约定的 token bucket 决定。两个层合计每 token H2D 约 `2×3×256×2=3072` bytes/rank，整节点24 KiB/token；4096 tokens 两层约12 MiB/rank，host 双缓冲约24 MiB/rank。NPU gather 后全 heads 为每层12 KiB/token，HCCL all-gather 流量另外计入。

### 8.3 首版完整 graph 路线：图外准备，图内消费

查表只依赖 token IDs、历史、mask 和固定 hash 参数，不依赖层间 hidden state；因此两个 Engram 层的 rows 可以在整个模型 forward 前准备。只有 projection/gate 依赖 hidden state，留在 NPU graph 内。

```text
本步 scheduler metadata + 真实 token IDs + 每请求过去3个 token
    ↓ 图外：校验请求身份/position，生成或取得 hash IDs
host CPU gather 本 rank 三个 heads → pinned host staging
    ↓ copy stream：H2D 写固定地址的 device_rows[layer1/layer14]
ready event
    ↓ graph stream wait ready（在 replay 前建立本步依赖）
model graph replay：layer1/layer14 读取 rows → TP all-gather
    → BF16 wkv projection → FP32 gate/norm → residual add
    ↓ done event
归还 staging slot；下一步可以复用
```

首版允许在 replay 前等两层都准备完，减少跨 graph 的 host 调度复杂度。必须在 **每次 replay** 的外层执行 prepare hook，而不是只放在首次 capture、dummy_run 或普通 model forward 内。

Graph 捕获时使用初始化为零的固定 rows 和有效 mask；捕获不能推进真实请求历史，也不能拿 dummy hash 当真实 ID 查越界。图内输入地址、dtype、shape 保持固定；真实 token 数作为固定地址 metadata/mask 更新。Host staging 可以轮换，但 copy 目标必须仍是该 graph 捕获的 device 地址；若轮换 device bank，则必须每 bank 分别 capture，不能替换 Python tensor 引用冒充更新。

CPU worker 不能在 H2D 完成前覆盖 pinned buffer；copy stream 不能在旧 replay 完成读取前覆盖 device rows。使用 `host_slot_free → gather_done → h2d_done → graph_done` 的明确事件/所有权协议，每个 batch 描述含 generation ID，避免请求重排后旧任务覆盖新内容。buffer/owner 必须存活到相关 stream 完成。

### 8.4 Token 历史与异步调度

Engram hash 必须逐位对齐：token normalization、compressed vocab99092、每层 RNG seed、奇数 hash multipliers、prime bucket offset、64-bit 运算、2/3/4-gram 与 pad/dead 区分。沿用参考语义编写 CPU reference 和 AscendC hash kernel；不能用近似字符串处理重建 token_map。

Prefill token IDs 已在 host，CPU 可一次批量 hash/gather 两层。decode 的最新 token 可能仅在 NPU，尤其 async scheduling 下 CPU token table 可能是 placeholder，不能直接读取它当真实历史。首版可关闭 async scheduling，或每 step 一次 D2H 同步取得所需真实 IDs/history/hash，再启动 CPU lookup；禁止每层逐 token `.item()`。

优化路线由 NPU AscendC 批量产生两层 hash IDs，单次 D2H 到 pinned ID buffer，event 完成后 CPU worker gather。该 D2H 依赖是实际成本，不能把未来 sampled token 的预取当成可以无条件隐藏的工作。layer14 能利用更长计算窗口，layer1 窗口较短，是低延迟瓶颈之一。

需要覆盖：chunk 开头过去3个 token、短 prompt/pad、batch 重排、prefix hit、SWA block 回收、请求抢占/恢复/取消、生成 token 的 commit、speculative accept/reject 回滚，以及 image spans 的 DEAD mask。历史归属 request ID + logical position，不能只按复用的 batch row 保存。Prefix/KV 传输并不自动带上 Engram 历史；启用相关功能前必须恢复真实 lookback，或明确拒绝不支持的组合。

### 8.5 性能阶段：piecewise 重叠

正确性基线通过后，把 layer1/layer14 消费位置作为必要分段边界：图外尽早启动两层 gather/H2D，segment 到达该层前等待 ready，再进入消费 segment。layer14 lookup 可与前13层计算重叠。必须核对 Ascend 的 `ACLGraphWrapper`、piecewise compiler、offloader hooks，不能把 CUDA 的 `eager_break_during_capture` decorator 原样使用。

首版 full decode graph 与后续 piecewise overlap 都算 NPU graph 兼容路线，分别报告端到端性能。若平台支持把 H2D 或 event 操作捕获入图，仍需用多次变化输入验证地址、host lifetime 与 event generation；首版不依赖该能力。

### 8.6 Engram 设备计算

按参考，rows flatten 后 wkv 输出 `(hc_mult+1)×hidden`，分成4路 key和1路 value；gate 对每一路 hc、沿 hidden5120 单独归一化，计算加权 dot，做 signed sqrt 后 sigmoid，再把 gated value 加回该路 residual。q_weight 与 k_weight 只以乘积使用，但融合前后 rounding 必须验证。

projection 先使用现有 BF16 Linear/Cube；gate/norm/residual 采用 AscendC AIV，规约用 FP32，保留 clamp、eps、mask 和 dtype cast 顺序。不能对4路合并求 RMS，也不能预取最终 gated hidden，因为 gate 依赖当前层真实 hidden。

## 9. CSA2、mHC 与缓存实现细节

1. 在模型初始化时由配置生成不可变 layer-role 表；验证每个消费者存在此前有效 source。源层注册实际 KVCacheSpec，消费者共享引用，不重复申请4/20份全局缓存。
2. **Compressor 重写**：以 V4.1 官方 reference 的公式和张量契约为依据，旧 Ascend compressor 仅用于定位调用接口，不复用其数值逻辑或复杂流水。矩阵乘法单独执行，详见下节。CR2 的 FP32 `kv_score` 和未闭合组 ring 跨 chunk 持久化；奇数起点/末尾、一个 token 的 chunk、页边界需与参考对齐。CR1 无同样的 pooling ring。
3. Compressor latent 的 main-cache insertion 和 indexer K projection 可以共享输入，但写入完成事件必须在消费前可见。RoPE 使用 source层/压缩位置的正确规则，不能将 decoder reuse 时重复旋转。
4. candidate producer 层20每步写 top2048 blocks；consumer 24/28/32/36 仅在有效候选与 causal mask 内取top512。验证候选不足、短序列、padding -1、score ties、block8尾部、不合法索引；graph 中不得把上步候选当本步输入。
5. 现有 `quant_lightning_indexer_v2` 的 mode1/2/3、CR1/2、PA_BBND/TND、cu_seqlens 组合逐项测；查 op_host 限制只是第一步，还需真实910B kernel结果与top-k reference。
6. 首版 attention 的 KV/index cache dtype 独立于权重量化选择：能证明正确的 BF16 baseline 优先。复用已有 INT8 cache/indexer 路径必须独立量化误差评估。FP4 main KV 是后期软件格式/算子任务，不将910B描述为原生MXFP4计算平台。
7. V4.1 mHC `attn_pre` 给本层 FFN，`ffn_pre` 给下一层 attention；第一层初始 pre-mix、Engram 插入位置和最后 hc collapse 对照 reference。`rms_norm_eps=1e-20` 不可被老V4默认值替换。
8. 主干 eager 保留完整语义后再评估 CED prefill 跳算和 SWA bounded replay。必须证明最终输出位置、chunked prefill、prefix-cache命中下与完整计算等价，不能按“20层encoder/20层decoder”直接跳过后20层。

### 9.1 Compressor 重写的模块边界

用户进一步明确旧 compressor 不正确且逻辑过于复杂，因此按以下边界重新实现，不采取在旧 kernel 中叠加 V4.1 分支的方案。

```text
hidden
  → 独立 projection GEMM（现有 Linear/Cube；必要时单独 CATLASS）
  → CR1 BF16投影 / CR2 FP32 kv_score
  → 新 AscendC CR1 / CR2 压缩与归一化
  → BF16 latent
       ├→ 独立 RoPE / main-cache store
       └→ 独立 indexer-K projection → k_norm → RoPE / index-cache store
```

- **GEMM 不进入 compressor kernel**：projection 权重、matmul tiling、Cube流水归独立矩阵乘组件；官方reference中CR1输出BF16，CR2投影/score输出FP32，不能通过先降成BF16再升FP32假冒CR2契约。CR2可利用BF16源权重/输入的Cube乘法与FP32输出，但必须与FP32 reference比较，而不是假设累积精度相同。
- **CR1 单独路径**：输入 BF16 `[T,512]`，按有效token做reference要求的RMSNorm，输出 BF16 `[T,512]`；不创建不需要的CR2状态，也不混入CR4/128逻辑。
- **CR2 单独路径**：输入 FP32 `[T,1024]`，明确分开512维KV与512维score；沿两个token、每channel独立softmax加权合并。按官方reference先把FP32 pooled结果cast回BF16，再执行RMSNorm，输出闭合组latent。未闭合组持久状态、跨chunk读入与本chunk写回显式处理。
- **Cache insertion 独立**：位置、slot mapping、RoPE、存储量化由单独算子负责。compressor 不承担main/index双cache布局、Top-K或candidate逻辑。
- **两类简单任务**：每request一个边界任务，先读旧ring并处理chunk首部跨界pair，再写ring尾部；chunk内部完整pair任务只读raw输入，在各AIV间分发，不碰ring。无需跨核同步；禁止将一个长prefill请求的所有pair串行压在一个AIV上作为性能交付版本。
- **不预先做CV融合、多流或跨核flag协议**：先用纯AIV实现压缩与norm，只保留局部必要同步。后续优化必须有profile证据且保持GEMM独立这一边界。

独立测试projection输出、CR1、CR2闭合组、CR2状态写回、norm、RoPE/cache store各阶段；再做链路测试。覆盖空batch、长度1/2/3、奇偶start position、连续多chunk与一次性prefill等价、多个request交错、slot回收和graph重复执行。所有状态在dummy capture期间使用隔离buffer或明确重置，不能污染真实序列。

**已核出的CUDA/reference差异**：参考 `inference/model.py::Compressor` 的CR1使用BF16 projection，CR2在pooling后先转BF16再norm；当前vLLM CUDA在 `attention.py::_run_parallel_input_projections` 使用FP32输出GEMM，`common/ops/fused_compress_quant_cache.py::_store_latent` 直接对FP32 pooled求norm，省略中间BF16 rounding。首版以官方reference数值语义为准，CUDA实现作为性能/误差对照；任何省略cast的优化需独立评估后再启用。

**Chunk/state的完整契约**：官方简版reference在非零 `start_pos` 分支只处理单token decode；任意chunk需另写等价reference。服务ring为 `[blocks,capacity,1024]`，上游capacity至少8并随speculative长度扩展，不应只保存一个pending token后声称支持rollback。无推测基线可限制功能，但状态接口保留capacity和逻辑position。测试比较有效状态语义，不要求与简版reference的两行物理buffer布局相同。

CR2有效latent写在pair末token对应的输出行，shape固定 `[T_bucket,512]`；无效行建议置零，cache store仍必须按有效mask屏蔽。RoPE采用组首位置 `floor(position/ratio)×ratio`；indexer路径不能漏掉k_norm。dummy capture与真实ring隔离，graph replay更新position/slot时必须重新计算闭合mask。Compressor只运行于KV source层2/8/14（CR2）和20（CR1），性能预算不能按40层重复计算。

独立projection建议契约：CR1 `BF16[T,5120] × BF16[512,5120]ᵀ → BF16[T,512]`；CR2将wkv/wgate沿输出维合并，`BF16[T,5120] × BF16[1024,5120]ᵀ → FP32[T,1024]`。已有arch22 `csrc/common/include/kernel/l0c_to_gm_iterator.h` 的 `Fixpipe<float,float>` / `NoQuant` 可参考，但必须独立验证数值与性能，不能凭API存在宣称完成。

## 10. 代码落点与开发约束

遵循仓库 `AGENTS.md` 的 plugin/patch/inheritance 模式。该文件要求不直接新增模型文件；因此先扩展现有 DeepSeek V4 模型容器，以独立 V4.1 subclass/组件区分行为，配合最小 patch 与注册。当前仓库虽已有 `models/`，不据此忽略该约束。

| 落点 | 计划修改 |
| --- | --- |
| `vllm_ascend/models/__init__.py` | V4.1 NPU 注册，避免落入上游 NVIDIA 默认入口 |
| `vllm_ascend/models/deepseek_v4/model.py` | 在现有文件中增加 V4.1 specialization；mHC、layer role、Engram消费、完整权重映射，与V4路径隔离 |
| `vllm_ascend/models/deepseek_v4/vl_model.py` 等现有文件 | V4.1视觉wrapper、config和processor复用/差异 |
| `vllm_ascend/patch/platform/`、`patch/worker/` | 必要的配置能力校验/平台分派 patch；准确限定model type与NPU，避免全局放开CUDA限制 |
| `vllm_ascend/worker/model_runner_v1.py` 或 `worker/v2/` | 选定一个首版runner，添加每步图外Engram准备hook与真实lookback；第二runner之后单独验收 |
| `vllm_ascend/compilation/acl_graph.py` | 尽可能复用既有边界；确有需要才扩展 per-step ready/done 协议 |
| `vllm_ascend/attention/`、现有 compressor Python封装/indexer | V4.1 source映射、candidate buffer、metadata与cache specs；compressor封装改为独立GEMM + 新CR1/CR2算子 + cache store |
| `vllm_ascend/ops/` | Engram host manager封装、CPU gather绑定、AscendC算子调用；生命周期由runner/model实例持有 |
| `vllm_ascend/quantization/methods/wna16/w4a16.py` | signed scale验证、bounded repack、small/large batch dispatch |
| `vllm_ascend/quantization/configs/` | 混合BF16/W4A16配置、忽略列表、converter格式对接 |
| `csrc/moe/w4a16_moe/` | 从个人分支迁移kernel并补288尾块、BF16精度、tiling |
| `csrc/moe/hc_pre/`、`hc_post/` | 必要的Single-Pass mHC算子支持 |
| `csrc/attention/quant_lightning_indexer_v2/` 等 | 现有candidate/CR1/2能力验证后补缺陷，不重复造轮子 |
| `csrc/attention/` 新V4.1 compressor目录 | 重写纯AIV CR1/CR2压缩/norm与明确的状态契约；不复用旧compressor逻辑、不包含矩阵乘法 |
| `csrc/` 新Engram算子目录 | AscendC hash（如需NPU路径）、gate/norm；CPU lookup用C++，不把CPU代码称为AscendC |
| `csrc/torch_binding.cpp`、`torch_binding_meta.cpp`、各CMake、`csrc/build_aclnn.sh` | 完整注册、Fake/Meta、alias/mutation语义、arch22构建列表 |
| `tools/` 或 `examples/quantization/` | 流式转换、inventory/校验、断点manifest、抽样误差报告 |
| `tests/ut/`、`tests/e2e/`、`benchmarks/` | 数值/状态/graph测试、8卡E2E、性能结果 |

上游 V4.1 包的 `__init__` 会触发 NVIDIA 导入，不能以为导入 `common.*` 一定无CUDA副作用。NPU路径尽量复用真正平台无关API，必要时在插件中带出处移植小型公共逻辑；不复制整个CUDA模型树。

AscendC技能采用 `ascendc-op-dev`。API、模板、硬件行为从当前CANN接口/实现和 `Ascend910B3.ini` 核对。算子改动后必须全量构建并确认新安装产物，再进行NPU测试；优化同步前复读事件配对规则。新增配置尽量放现有结构化配置，若需要环境变量集中定义于 `envs.py`，文档化范围/默认值；patch与runner扩展在变更集中提供架构说明。

## 11. 验证矩阵与性能度量

### 11.1 三层数值基准

先分清硬件迁移误差、权重量化误差和KV量化误差，避免最终logits不一致时无法定位。

| 基准 | 内容 | 用途 |
| --- | --- | --- |
| R0 | 官方数学语义，原始FP8/MXFP4正确解码后的高精度reference | 验证模型结构、packing、RoPE、mHC、Engram与CSA2 |
| R1 | 相同数学语义，routed权重替换为目标INT4反量化值，KV先BF16 | 隔离RTN损失；评估各层hidden、logits与模型质量 |
| R2 | 910B kernel与完整NPU模型，使用相同目标权重/KV格式 | 与R1比对实现误差；再比较eager、graph、优化前后 |

无CUDA硬件时可用CPU分块reference验证单层/算子，不能声称已运行CUDA基准。全模型R0精度评估可采用可获得的官方结果或外部CUDA运行，但报告必须区分自己实测与外部数据，且使用同一tokenizer、prompt、上下文和解码配置。

### 11.2 必需测试

| 范围 | 关键输入与失败模式 | 验收 |
| --- | --- | --- |
| MXFP4解码 | 全16种nibble、低/高顺序、不同E8M0 exponent、跨group/block | 与独立reference逐值一致 |
| RTN/packing | 正/负dominant、tie、全零、负scale、q=-8、round ties、K轴group32、CT/native编码转换 | pack/unpack整数逐值一致；scale符号与reference一致；非法格式明确报错 |
| 转换器 | weight/scale跨shard、缺分片、截断文件、重复/缺key、异常中断、续转参数改变 | 不发布不完整checkpoint；重跑不重复转换有效分片 |
| W4A16 | hidden5120，inter288/2304，w13/w2，top6；B1至prefill级；空expert、不均匀路由、padding | 高精度ref误差与阈值分别报告，不能仅沿用旧128维toy测试 |
| Routing | sqrtsoftplus、bias影响选择但不错误混入输出权重、renorm、scale1.5、视觉mask | expert IDs及权重对齐reference，接近tie情况单独报告 |
| Compressor | 独立GEMM、CR1/2、跨chunk pair、ring回收、norm/RoPE/store | 一次prefill与各种chunk切分结果及最终状态一致 |
| CSA2 | 全部source/reuse关系、top512、candidate2048×8、短/长context、future/pad mask | 不读无效cache、不泄露未来token；top-k和值对齐reference |
| mHC | 首层pre-mix、attn→FFN→下一层顺序、四路规约、Engram前后、末层collapse | 逐阶段比较，确认未沿用V4 learned hc_head |
| Engram hash | tokenizer归一化、pad/dead、图像、过去3个token、request重排 | hash IDs整数逐位一致 |
| Engram offload | 每步改变token/hash、重复row、无效ID、大小bucket切换、取消/回收、NUMA | rows与CPU直接lookup逐值一致；无stale result、无泄漏 |
| Graph | capture后至少100步变化输入；不同batch有效长度；host/copy线程人为延迟；多流事件复用 | eager/graph输出一致，buffer未提前复用；不能只replay同一输入 |
| 8卡分布式 | TP8、可选EP8、全部rank加载映射、HCCL、head gather次序 | 与单层分片reference一致，无重复scale/重复routing权重 |
| 模型/服务 | 文本长短prompt、持续生成、prefix/chunked、vision、DSpark接受/拒绝、tool/reasoning | 每个启用能力有E2E结果，unsupported组合显式处理 |

浮点算子阈值在reference和真实分布基线后冻结：同时记录max/mean absolute error、NRMSE、cosine；低幅值场景不只用max relative error。RTN理论损失与kernel额外损失分别统计。graph路径原则上与相同kernel eager结果一致；如非确定性规约引入差异，需解释并设紧阈值。

整模型量化质量至少包括 held-out 文本perplexity/NLL、固定prompt teacher-forced logits/top-k、长上下文检索/理解，以及代表性推理/代码任务。贪心生成文本一致性是辅助观察，不替代teacher-forced对齐；只要早期token不同，后续自由生成误差就不能用于逐层定位。

### 11.3 性能实验

统一记录模型/转换manifest hash、8卡拓扑、CANN/PyTorch版本、graph模式、batch/token bucket、上下文、KV dtype和并发。输入长度覆盖128/2K/8K/32K/128K，输出长度128/512；1M作为容量与精度独立阶段，不提前承诺吞吐。

测量TTFT、TPOT（P50/P95/P99）、prefill tokens/s、decode tokens/s、端到端请求吞吐、HBM峰值、host RSS/Pinned/NUMA带宽、H2D/D2H带宽和HCCL时间。warmup/capture与steady-state分开；Engram随机访问冷/热情况分别测。

使用msprof重点回答：layer1等待host多久、layer14拷贝被隐藏多少、host lookup/H2D/HCCL各占多少；小batch AIV是否比GMM快；288尾块和BF16累积成本；candidate过滤是否减少实际索引工作；重写compressor的projection和向量部分各占多少。每次只改变一个优化变量。

## 12. 风险、退出条件与后续行动

| 风险 | 已有证据 | 处理 |
| --- | --- | --- |
| 旧kernel不能直接用于TP8 | 旧tiling要求128对齐，实际局部维288 | 真实shape测试优先；补tail或对照EP，不隐藏padding成本 |
| signed INT4编码误读 | 现有loader减8，而原生kernel用two's-complement | 格式manifest + 全码点roundtrip，转换与repack明确分界 |
| 大表pin失败/NUMA过慢 | 约366GiB host，单进程46GiB全区写入/H2D已通过 | 继续验证8rank同时pin与NUMA随机访问，保留明确的bounded-pinned模式 |
| graph使用旧rows | Python CPU操作不会随replay自动执行 | 每step外层hook与ready/done事件；变化输入压力测试 |
| async历史是placeholder | 官方Engram说明V1 CPU生成token表的限制 | 真实NPU history批量D2H或先关闭async，逐项恢复 |
| 上游CUDA导入/配置限制 | V4.1默认NVIDIA入口、EngramConfig限制CUDA | NPU注册与最小平台patch；启动阶段import/API测试 |
| mHC/CR2语义偏移 | V4.1 pre-mix延迟和source/cache关系变化 | 分阶段reference，不以旧V4结果为真源 |
| 旧compressor不可信 | 用户明确指出旧实现错误且过于复杂 | 独立GEMM，重写简单CR1/2 AscendC，不修补旧流水 |
| 64GiB装载峰值 | repack可能展开INT32，双layout可能翻倍 | 分片直接加载、小块repack、峰值测量，避免完整权重双份 |
| 新旧HTTP依赖无法同时满足 | FastAPI约束无交集 | 隔离开发环境，API验证后在fork调整约束，单独验收serving |
| 模型下载未完 | 缺47/48 Engram分片 | 其他工作持续推进；全表验收严格等完整文件/checksum |

当前实现优先顺序：完成真实runner/cache/model构造与文本E2E；补齐全表加载后进行8卡graph与质量验收；随后恢复vision及DSpark，最后整机profiling。converter、算子和组件级证据见以下更新。按用户授权，将接口已确定的独立AscendC算子交给sub-agent并行开发；后台编译不阻塞文档、CPU reference或其他独立工作。

### 12.1 当前实测记录

- 已创建个人fork开发分支，原工作树起始干净。
- 已完成前46个分片header/文件长度盘点、按dtype归类和容量估算。
- 已完成layer0/expert0的w1/w2/w3真实RTN抽样，结果见6.2。
- 原容器torch-npu检测到8卡；小BF16 host tensor `pin_memory()` 成功，non-blocking H2D后逐值一致。
- 基础 `torch.npu.NPUGraph` 固定输入缓冲分别写2/7再replay，输出分别为3/8，检查通过。这只是基础能力检查，尚未验证完整Engram host lookup或多流graph。
- 另完成100步合成staging/graph实验：每步变化CPU lookup IDs与有效token mask，双pinned host buffer、独立copy stream、ready/done event、固定device rows地址，再replay并逐值比较，全部通过。每步读取结果会同步，所以此实验验证基本多流顺序，不证明多个请求在途重叠或完整Engram集成；日志 `/tmp/deepseek-v41-staging-graph.log`。
- Compressor独立sub-agent设计核查已完成，确认CR1/CR2 rounding差异、两类AIV任务、ring/rollback、k_norm和RoPE组首位置；建议已纳入9.1。
- vllm-ascend完整editable构建/安装已成功：`.venv`内版本 `0.1.dev5255+gb49962987`，生成 `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`、`libvllm_ascend_kernels.so` 和custom OPP；import确认 `vllm_ascend.__file__` 指向当前个人fork。系统旧editable未被卸载。
- 新vLLM首次metadata检查发现缺少 `setuptools_rust`，补齐后empty editable安装成功；`vllm.__file__`确认指向 `sources/vllm`，`current_platform.device_type == npu`，`VllmConfig`与新Ascend `.so`可导入，`npu_quant_lightning_indexer_v2`已注册。candidate扩展的底层接口实际还包含`npu_quant_lightning_indexer_v3`，后续按schema接入，不能只依据名称后缀选择。
- `pip check`尚不通过：除原容器profiler/OpenCV问题外，当时开发环境另有FastAPI/Starlette及huggingface-hub版本冲突。现隔离环境已更新FastAPI0.136.3、Starlette1.6.0、huggingface-hub1.31.0，plugin fastapi约束同步；TestClient health和HF HTTP接口导入通过。系统profiler/OpenCV冲突仍待单独处理，完整HTTP模型服务尚未验收。
- 本文Markdown lint通过；未运行完整模型、未转换全部权重、未验证视觉/DSpark/1M。

### 12.2 交付物

最终变更集应包含本计划、可续转的转换工具与格式说明、重写compressor、W4A16 AscendC、Engram host manager与graph集成、V4.1模型适配、对应测试与benchmark、经过实测的8卡启动配置和环境锁定记录。最终代码在个人fork分支，提交遵循仓库sign-off要求；当前未向上游仓库推送或创建PR。

### 12.3 实现进度更新（2026-09-15）

- 转换工具位于 `examples/quantization/convert_deepseek_v41.py`。前3个真实分片完成后，使用独立逐group RTN参考检查了MoE packed codes/scale，dense BF16反量化逐值一致；随后后台续转所有已下载分片。转换器9项UT通过，仍缺源47/48，尚未发布完整模型config/index。
- 新 `CompressorV41` 完整构建安装后通过42项NPU测试，包括变化metadata的graph replay、分块、ring及rollback。38个graph微基准（每图unroll256次，5轮）相对正确批量PyTorch参考几何均值13.4138x，最大轮间中位波动0.2843%；只计vector/ring/norm，不含GEMM、store或8卡通信。完整模型尚未接通。
- W4A16复用个人分支 `origin/deepseek-v4-w4a16-tp` 的 `274dd30c3230804e35b7985e288dacbde3326ece`，修复I288尾块、FP32累积、FFTS启动模式及B32 buffer复用同步。150项UT及24项NPU测试通过。仅明确decode且B≤4/TP8/H5120/I288/E384/top6的BF16 group32路径允许实验dispatch；prefill、B≥8或不支持组合继续CANN GMM。`enable_w4a16_decode`默认False，等待整机验收。B4热点native median/P95为240.96/250.98µs，CANN为340.32/351.02µs；轮间仍有3–12%波动，不能宣称全性能门槛通过。移除CANN可省略的全零offset，每rank减少约3.955GiB常驻内存。详见 `benchmarks/deepseek_v41/W4A16_STATUS.md`。
- `ops/engram_offload.py` 实现TP head分片直接加载、双pinned staging、固定NPU rows、copy/compute事件及runner生命周期协议。8项host UT、3项NPU测试通过；后者覆盖64步在途eager/graph、bucket4/8、零token/padding、多层rows及稳定地址，循环中不读回结果。单进程46GiB pinned分配、全区写入、随机抽样及H2D验证已通过；8rank同时全表pin与真实runner接入仍待验收。
- `ops/engram_hash.py` 实现图外stateless host hashing，调用者提供真实lookback而非async占位历史。5项UT通过；真实tokenizer构建出99092压缩词表，prime和multipliers与模型参考一致，12组分块/图片mask/重算hash逐值一致。此结果不代表已解决runner实际token获取。
- Engram post-wkv独立AscendC门控正在统一完整构建，严格遵循模型FP32乘法结合顺序；投影GEMM保持独立，NPU精度/性能尚待新产物验收。

### 12.4 新增集成证据与构建一致性

- 发现旧ACLNN构建保留 `*_src_copy.done`，导致完整pip安装仍复用旧kernel编译副本。`csrc/build_aclnn.sh`现清理生成的 `build/output/build_out` 后完整重建。r7中间编译副本已与Engram新源码吻合，`.o` SHA由 `a51d625f...` 变为 `c7b59c27...`；必须安装结束后重跑精度，旧binary测试不能用于判断Div修复效果。
- `AscendIndexerV41Ops`支持CR1/2、paged INT8 K、source blockmax候选和consumer top512；15项CPU/NPU/graph测试通过。27项graph性能中23项延迟达标，全部稳定性达标，4项B1仍不达标。当前consumer仍全量QK再mask，不能把候选过滤描述为QK算量减少。报告 `benchmarks/deepseek_v41/indexer_v41/report.md`。
- Delayed mHC复用现有AscendC：5项NPU测试及真实layer0/1/2/20/39控制权重对照通过。参数保持FP32，但既有HcPre Cube内部HF32计算的模型质量影响仍需整机评估。Decoder组件通过1项对照测试，其中attention/MoE是明确的stand-in。
- `CompressorV41`独立GEMM、环形cache绑定及动态device metadata完成，3项NPU组件测试通过；runner环形cache分配/reshape回归通过。未宣称完整scheduler/graph模型已接通。
- `DeepseekV41AttentionProjections`完成BF16 Q/KV输入投影、低秩Q归一化、FP32相邻对RoPE及逆RoPE分组输出投影。CR0/1/2三项TP8局部尺寸NPU/变化位置graph测试通过，output all-reduce在测试中用identity替代；不等于8卡attention验收。
- `EngramRequestHistory`以request ID隔离真实执行历史，完整prompt可用于prefix命中；执行draft可覆盖并回滚截断，缺失/placeholder历史拒绝。已补CPU测试与接入说明，最终真实IDs/positions批量D2H及runner调用尚待连接。
- 主attention正在验证现有910B `sparse_flash_mla` 的隐式SWA128+显式CSA512路径。BF16 cache作为数值基线，与官方原生低精度cache不同，须报告质量和容量差异。该阶段尚未注册V4.1顶层模型；最新注册与集成进展见12.5。完整权重加载、文本/视觉/DSpark及整机profiling仍未完成。

### 12.5 模型接线、数值根因与当前验收边界

- Engram r9生产kernel完整安装后17项严格NPU测试全部通过。诊断确认基础 `Rsqrt(1)=0.998046875`，两次归一化导致系统偏差；两处改为 `Sqrt + Div`，sigmoid末端保持Div，未放宽原NRMSE 2e-4门槛。8个shape性能门槛全部通过，最大NRMSE 1.24e-5；graph T1/T8/T64/T1024分别3.55/5.87/19.34/231.87µs，只计post-wkv gate，详见 `benchmarks/deepseek_v41/engram_gate_910b.json`。
- Compressor原42项BF16测试不能排除同类Rsqrt系统误差。已将norm改为Sqrt+Div，并新增8个CR1/2×幅值严格归一化回归；r10完整构建安装后54项参考/NPU测试通过；38项主graph矩阵及12项odd-closing decode矩阵均满足原性能/稳定性门槛，相对正确PyTorch参考几何平均12.97×/14.73×。只计vector/ring/norm，不含独立GEMM或整机成本。GEMM仍为独立调用，不改变kernel ABI或tiling。
- 主attention已完成24项NPU数值/graph测试；CR0约22–24µs、CR1/2约38–41µs，仅局部算子计时。完整独立projection→compressor→indexer/cache→joint attention链路6项NPU/变化输入graph通过。registered cache自动解析另有4项NPU测试通过：真实planner→runner views→bind→metadata→source/consumer自动解析→变化物理页graph；仍不等于完整runner接通。
- Cache spec、32KiB统一page planner和runner views已实现。planner及相关回归67 passed/1 skipped，runner allocator/reshape 21 passed，main/index/SWA scatter 57项NPU+6项CPU通过。仅支持block32、CR2 ring capacity8及LBNHC/LBHNC紧凑布局；环形状态仍连续，不伪装支持任意page stride。
- 已在现有 `models/deepseek_v4/model.py` 新增V4.1 backbone与LM类，并注册NPU文本入口，避免导入官方默认NVIDIA实现。权重loader与metadata联合38项UT通过，包括真实Ascend线性层TP切片、fused projection、expert packed/scale/shape callback与vision附属权重过滤。缩小到3层/E8、保留production H5120/I2304的真实TP8 constructor已通过8rank验收，确认W4A16 factory、完整参数shape/dtype及6类cache注册；未做forward。修复此前按EP通信组大小误算TP-only专家数量的问题，新增EP/TP CPU回归。实际完整checkpoint加载及文本生成仍未验收。
- Engram实际runner hooks已接：最终device IDs/positions/query bounds批量D2H→request history/hash→双pinned H2D→wait_ready→model/replay→mark_consumed。history与runner helper回归通过；修复旧sanitizer将未补齐-1改0的问题，真实placeholder必须拒绝。全execute和dummy capture路径仍需E2E。
- Engram Runtime组件1项NPU测试通过，覆盖prefix/mask/draft rollback与稳定地址。8卡两层合成host表→pinned DMA→HCCL head gather→32步变化输入graph逐位一致；该测试不含真实全表、wkv/gate或完整模型，不作为吞吐证据。见 `benchmarks/deepseek_v41/host/engram_tp8_smoke.json`。
- 新增NPU/V4.1专用Engram配置resolver；只接受CPU pinned TP shards，拒绝尚未接通的跨DP/shared、CP、DBO、EP、PP与speculation组合，其他模型/平台沿用原逻辑。27项配置UT通过，真实EngineArgs启动配置通过，V41 block32不再被通用chunked-prefill逻辑改成128。Host factory另21项UT通过，缺表/错shape在任何pin前拒绝；默认lazy加载，拒绝会私有整表读取的eager加载。模型/配置/loader/history/runner helpers合并133项UT通过。后续能力实现后应逐项恢复，不将临时限制写成硬件限制。
- 权重转换已完成当前已下载46片，缺47/48 Engram源文件，任务已正常退出。仅完整转换并校验后发布目标config/index；当前没有可用于整模型加载的完整目标checkpoint。

### 12.6 真实 TP8 runner 请求验证

- 三层/E8、生产H5120/I2304、真实host Engram小表的完整LLM/EngineCore/scheduler/8worker请求已通过eager和FULL_DECODE_ONLY graph。40/48-token prompt按32-token预算分块，两个并发请求各生成4token，再执行一个复用前缀的请求生成4token；全部12个token和selected logprob在独立eager/graph进程间完全一致。
- 每rank有9次Engram prepare；rows/mask在capture前后与请求结束保持相同地址，无未消费的step。每rank捕获2张图，日志确认实际replay。该测试使用dummy device weights并显式初始化packed INT4，不是质量或性能验收；E8也不触发生产E384 native decode优化。详见 `benchmarks/deepseek_v41/runner_tp8/report.md`。
- 真runner暴露并修复两个集成问题：新上游metadata已移除的三个constructor字段改由Ascend子类保留，unpadded保留新上游字段；plugin cache binder调用层自定义binder，修复compressor raw 4D到3D绑定。19项metadata回归、2项binder CPU回归及4项registered-cache NPU测试通过。
- 同一runner完整40层/E8的eager与graph也已通过，保留两个Engram、CR1/CR2切换及全部source/consumer布局，全部12个token及selected logprob完全一致。随后1152/1160-token长prompt对比也通过，每rank44次prepare，覆盖SWA淘汰与CR1/CR2超过top512的选择压力；候选块真正淘汰仍需超过16384-token验证。
- 三层/E384的CANN eager、native decode eager、native decode graph均通过，token及selected logprob完全一致。每rank确认18次eager native dispatch或6次native capture，prefill保持CANN。补齐V41 host执行分类metadata及hybrid metadata查找，68项CPU回归通过；runner另外禁止未完成prompt的单token尾部误用decode graph，2项runner回归通过。仍是合成device权重，不能据此默认启用或声称整机性能提升。
- 正在通过正常safetensors loader加载前三层/E384真实转换权重，临时测试目录仅用小Engram表替代缺失源表；不修改或发布未完整转换的目标checkpoint。完整40层/E384真实权重、全表pin及整机性能仍待验收。缺失47/48源分片的临时文件自17:59 UTC后未见增长，已询问下载进程状态；另已询问NPU7外部34GiB占用何时可释放，未操作该进程。

### 12.7 真实权重、NUMA与数值验收的新证据

- 三层/E384的真实转换device权重通过标准safetensors loader完成TP8装载及请求执行，每rank装载约3.4934GiB；Engram仍为小型合成表。CANN eager、CANN graph及native graph均生成相同12个token，但selected logprob仍有差异，最大约0.27；CANN graph自身也出现同量级差异，不能全部归因于native。完整数值一致性尚未通过，native默认关闭。
- 关闭prefix后，同一eager实例重复相同batch仍有最大0.061671的selected logprob差异。逐层trace显示首次prefill的tokens/positions及第一层input norm逐位相同，首差在第一层attention输出（NRMSE0.004062），早于MoE、Engram及CR2。正在细分projection、sparse attention、wo_a及wo_b归约前后；未据此放宽精度门槛。
- 后续全8rank trace确认：两次相同prefill的各rank wo_b本地GEMM以及FP32求和逐位一致，BF16 HCCL归约后首次出现差异（NRMSE0.004758）。该轮默认归约的重复请求logprob最大差0.0640504；使用HCCL_DETERMINISTIC=strict后token/logprob逐位一致。继续验证无trace的eager/graph，未修改产品全局默认或声称确定性模式无性能代价。
- 无trace的真实三层CANN eager/graph确认完成：strict HCCL、关闭prefix条件下，两模式12个生成token及selected logprob全部逐位一致，各自重复batch也逐位一致。正在补native decode graph复测，整40层/全Engram与模型质量仍未验收。归约证据及重放方法见 `benchmarks/deepseek_v41/ATTENTION_REPEAT_NUMERICS.md`。
- 实际请求捕获的三层MoE decode输入已独立对照：CANN重跑与捕获输出逐位一致，native/CANN NRMSE约0.50–0.52%。native保留FP32投影/激活与路由累加，CANN在反量化、两次GMM及routing使用BF16边界；native与FP32合同更接近。未发现scale布局或重复路由加权问题，不为追随低精度舍入而降低native精度。
- Engram新增可选显式NUMA路径：mmap→mbind→流式填目标head→MAPPED host注册。55项CPU、7项NPU加载/H2D/graph测试通过；配置/工厂另142项CPU回归通过。TP8真实三层graph逐rank查询到的页位置均匹配[6,7,4,5,0,1,2,3]，整个请求期间地址与位置稳定；不等于366GiB全表容量/吞吐验收。默认保留原torch pinned allocator，可用additional_config.engram_numa_nodes选择新路径。
- 独立全容量验收随后通过：8rank/16owner同时驻留393227699200B（366.221833GiB），该历史运行实际NUMA为[6,6,4,4,0,0,2,2]（以原始records.node为准）；16个完整VMA页数/NUMA归属、144行首中尾DMA、160次变化hash/DEAD/padding的offload+graph全部正确；16次注销及8进程exit0，无清理错误。注册中位9.688秒/约22.89GiB，设备reserved每rank6MiB。使用完整真实shape和合成数据验证容量，尚不代表真实全表装载或整机吞吐；见 `ENGRAM_FULL_CAPACITY_RESULT.md`。
- 真实模型配置保留vision router bias后，发现runner将mm_prefix_range写入frozen compressor metadata。现只向声明该字段的attention metadata写入，8项真实common metadata回归通过；各backend仍通过common.mm_req_doc_ranges接收范围。
- V41单个image ID路由修正：仅129264使用vision bias，V4的五ID合同不变；真实配置→router及既有回归136项通过。独立V41图像processor组件43项CPU对照通过，无NVIDIA模型包依赖，尚未注册MM入口。
- 独立vision tower/aligner组件22项CPU、6项NPU测试通过；但完整32层真实权重对发布CPU参考的tower NRMSE为11.67%、aligner为6.94%，未通过冻结门槛。正在缓存逐层oracle定位，未放宽门槛、未启用视觉请求。torch NPU rsqrt单独误差约1e-7，并非此前AscendC基础Rsqrt偏差。

### 12.8 显式图片路由与后续验收

- W4A16继续复用已有个人分支，仅decode优化，prefill保留CANN。真实三层strict native graph与strict CANN生成token相同，selected logprob最大差0.003904；native自身FP32 AtomicAdd重复次序存在小量变化，独立100次eager+100次graph均满足既有算子精度门槛，不宣称逐位确定性。详见 `W4A16_REAL_NUMERICS.md`。
- 真实照片经官方processor的32层vision tower/aligner通过原门槛，NRMSE分别0.0215005/0.0140770。先前梯度压力输入失败保留；CPU SDPA dispatch及FP32对照确认深层BF16敏感，未更改门槛。未注册MM wrapper完成266参数流式分派及51项CPU回归，实际MM请求仍待验收。
- 新增独立prompt image mask，经history最终位置选择→固定runtime buffer→runner→LM/decoder/MoE显式透传，保留raw IDs用于hash routing。生成token、literal sentinel、padding及非图片DEAD位置不会误走vision bias。76项CPU及2项runtime NPU graph测试通过；router/complete op另137项CPU通过，实际TP8待验收。
- 修复raw-token MM dummy capture输入签名，使用与运行时一致的IDs+固定embedding buffer；V41完整图片span不套用V4 leading padding。runner/metadata/V4 MoE等93项CPU回归通过。
- 显式mask接线后的真实三层E384/CANN/TP8 graph回归通过：strict HCCL、相同32/40prompt与NUMA、prefix off条件下，全部12token及selected logprob与旧strict结果逐位一致，同实例重复亦一致。独立router/complete op另2项NPU测试、24次变mask replay通过；真实MM请求正在以test-only注册进行集成。
- 真实照片MM eager请求首次通过：完整32层vision+三层真实语言权重E384、8rank各编码一次、完整189图片token和literal129264文本位置分别传递，输出finite。使用小型合成Engram、test-only registry；尚不宣称完整模型质量。此前暴露完整图片预算需要1024及fork/OpenMP线程池冲突，测试改为spawn；graph对照继续中。
- 补齐生产runner shutdown→runtime/manager shutdown：先完整device fence，再释放注册表；pending DMA或forward未mark_consumed也可安全终止，同步/注销失败可见并保留owner重试。64项CPU、2项真实NPU pending-shutdown测试通过，正常close仍拒绝未完成消费。完整TP8退出及真实全表注销仍单独验收，见 `ENGRAM_REAL_TABLE_ACCEPTANCE.md`。
- Indexer B1实验按gather/cast→独立BMM→score reduce拆分，生产dispatch未启用。r11内核通过但binding stride临时值导致末端C++编译失败；修复后的r12完整构建/editable安装成功，21项精度测试通过。首轮性能不达标；最初where测量包含其输入条件，不能将约151µs全归因于INT32 where。改用gather写出的原位置有效性掩码后，三长度中位约69.47/81.31/81.09µs，均有完整冻结/live/noise门槛通过记录；4097首次baseline噪声失败保留，独占复测通过。生产接入仍需单独验证。
- 真实图片MM graph请求通过：每rank实际replay6次、native capture3次、prefill继续CANN；8rank terminal shutdown RPC均成功。图像及后续literal-image-ID文本请求的token与eager一致，selected logprob最大差0.004553/0.0000656。完整model质量门槛不由该三层结果推导；父进程析构的5秒grace超时另通过显式shutdown验证。
- 当前HTTP依赖入口已完成只读CPU检查：FastAPI0.136.3/Starlette1.6.0、真实app/OpenAPI构建通过，未启动NPU或HTTP监听。无需为serving盲目升级NumPy，因现有Triton明确依赖1.26.4；实际服务请求与最终profiler运行仍待验证，见 `SERVING_DEPENDENCY_AUDIT.md`。
- 完整40层真实Engram及最终profiling仍受源47/48未完成限制；独立安全续传持续运行，当前代理每片约120KiB/s，不能用全容量合成表或组件结果代替真实全模型验收。

- 生产registry正式指向本地MM wrapper/processor后，真实照片TP8 graph+NUMA再次通过：每rank编码一次、replay6、native capture3、fallback capture0，完整189图像位置和literal ID文本均正确。所有host表NUMA页匹配指定节点，8rank终止RPC释放owner，EngineCore exit0；见 `mm_production_numa_graph.json`。此前test-only注册结果保留作历史记录。
- 生产wrapper/processor CPU回归87项通过。最大图片1024span/9189patch的真实32层encoder容量门槛通过，峰值allocated1.349GB；这是encoder组件容量结果，不是整模质量/HBM验收。

- 生产image-limit0路径亦通过：8rank视觉参数0、encoder调用0、graph replay6、all-false image mask，Engram正常注销且EngineCore exit0；见 `mm_production_limit0_graph.json`。真实localhost HTTP serving开始验证，尚不以app构建检查代替HTTP请求结果。

- 候选indexer可选接线完成：`additional_config.enable_indexer_candidate_decode=False`，仅CR1 consumer/B1命中，model按max_model_len预分配workspace。158项model/config/wrapper CPU回归及33项candidate CPU/NPU测试通过（含动态graph及未prepare/B2/T2回退）。40层/E8合成权重、1152/1160prompt、prefix+NUMA的TP8 graph通过，每rank candidate capture4，全部12token/logprob与历史native基线逐位一致，同实例重复一致，8rank释放且EngineCore exit0。真实整模质量和E2E性能门槛未由此替代，默认保持关闭。
- 标准`vllm serve`、生产registry/default worker的文本completion、SSE及图片chat HTTP请求全部通过；r1退出默认shutdown_timeout=0导致内部强杀EngineCore和共享内存警告，严格保留为`failed_cleanup`，不以父进程exit0掩盖。下一轮显式shutdown-timeout30并采集三层HTTP timeline，结果尚待验证。

- HTTP r2显式shutdown-timeout30后，8worker正常退出且无强杀/资源泄漏；输出handler在MPClient teardown后打印EngineDeadError，日志保留。实际8rank profiling raw采集成功，worker daemon内自动export失败，官方离线analyse于13.197s完成8份17MB级timeline。每rank6次graph执行/18次W4A16/8次Engram gate/8次compressor，重复图像命中cache未含视觉塔；这仍是三层fixture，不替代最终整机profile。900个raw/export文件及checksum保存到workspace artifacts，见 `HTTP_PROFILE_RESULT.md` 和 `docs/performance/deepseek_v41_910b.md`。
- 已启动一次性后台转换续接watcher（初始PID3204064，log `/tmp/v41-conversion-tail-watcher.log`），只等最终发布的47/48源文件，检查size后串行复用原转换器，任何异常停下，complete+最终index/config后退出；9项CPU测试通过。用户已恢复原下载；独立恢复器因检测到源前缀变化而安全退出，保留已有suffix，不再重启或触碰用户下载临时文件。

### 12.10 真实权重逐层验证发现的修复与全模型准入

- 源47/48已发布，转换watcher已自动续转；完整manifest/config/index发布前不启动完整模型。
- 真实DSpark三层E128/top3的独立stage oracle发现target/draft共用的`wo_a`布局错误：Ascend loader已转置为`[groups,width,rank]`，旧投影却按原shape重读。修复`da6da081a`直接使用正确布局；修前三层×八rank共24项失败，修后context9、33、129各712项全部通过，原门限不变。参考输入为实际各stage输入和合成target辅助状态，不替代整模型质量。详见`benchmarks/deepseek_v41/WO_A_LAYOUT_CORRECTION.md`。
- 历史target eager/graph一致性只能证明当时实现的一致性，不能证明上述投影正确；target、完整真实权重质量和最终性能须在修复后重验。
- 初始draft输入kernel补齐页表逻辑宽度和DCP ownership读取mask；45项NPU测试通过，包括拒绝、切片table和变化长度graph，提交`1e10173ae`。最大上下文的RoPE、attention可见性、proposal有效性仍需完整衔接，DSpark生产准入保持关闭。
- 完整headers预检得到target静态设备权重估计39.299GiB/rank、两张真实Engram共366.221833GiB pinned host。48.549GiB/rank启动余量为估算准入线，非实测峰值；卡7外部占用当前不满足要求。不会干预外部进程，同时准备低HBM的真实全表factory加载与行oracle验证。
- 真实全表factory及完整40层eager/graph验收脚本已准备，默认只执行CPU预检；记录48源headers、47已发布转换headers及cgroup v1容量。独立factory使用实际生产加载/注册路径，计划检查16个owner、144个源FP8→转换BF16→host行oracle和变化输入graph replay；整模graph要求同权重、backend和prompt的已通过eager参考。上述真实加载和整模运行尚未执行，准备记录不能视为验收通过。命令、容量推导和剩余门槛见`benchmarks/deepseek_v41/FULL_MODEL_REAL_TABLE_READINESS.md`。

### 12.11 完整权重发布与融合 lightning indexer 必需交付

- 2026-09-16最后分片转换完成：48个转换分片、`complete=true`、最终config/index已发布。用户已释放卡7，最新完整模型CPU准入没有blocker。真实Engram全表factory通过：366.22GiB/16owner同驻留、144行oracle、160次graph replay、400页NUMA采样、16次注销及8worker正常退出；详见`benchmarks/deepseek_v41/ENGRAM_REAL_FACTORY_RESULT.md`，不代表完整40层或HCCL验收。完整40层eager验收已启动。
- 用户明确要求融合lightning indexer作为最终必需项，已分配独立算子任务，优先参考`https://gitcode.com/G_W_E/ops-transformer/tree/qli_opt/attention`并优化现有完整QLI实现。此前gather→独立BMM→score→topk的拆分实验不能作为最终交付，`enable_indexer_candidate_decode`继续关闭。
- 融合必须保持候选块、分页、量化/舍入、head加权、top-k和NPU graph合同，消除拆分路径的候选BF16 bank及完整QK等大中间张量；同时验收数值、workspace/HBM以及完整selector的多轮median/P95，不能仅报告计算内核耗时。
- 当前允许先用已有完整lightning indexer完成真实整模测试；融合完成后必须接入重验质量、图重放及整机性能，不能因已有路径可运行就关闭此必需项。
- 完整40层/E384/真实366.22GiB Engram的TP8 eager首次通过：48分片生产加载、144行oracle、四组prompt两轮共32生成token、重复token/logprob完全一致、16次注销、8worker优雅退出和EngineCore exit0。结束时每卡Torch allocated39.5538GiB/reserved42.2188GiB，非HBM峰值。已启动相同checkpoint/backend/prompt的graph对照，详见`benchmarks/deepseek_v41/FULL_MODEL_RESULT.md`；raw-token测试不代表语言质量验收。
- 完整模型`FULL_DECODE_ONLY` graph对照随后通过：每rank实际replay26次，全部token和selected logprob与eager及自身重复逐位一致，device staging指针稳定；144行oracle、16次注销、8worker退出及EngineCore exit0均通过。结束时每卡Torch allocated39.5538GiB/reserved44.0977GiB；仍使用r12已有完整QLI，不替代新融合算子的最终接入验收。

完整模型自然语言 graph 后续验收：五项 exact-answer smoke 全部通过（算术、中文、英文提取、JSON 排序、4243 token 检索），每 rank 20 次 replay；144 行 oracle、16 次注销和正常退出通过。原始结果见 `benchmarks/deepseek_v41/full_text_graph_r1.json`，仍非广泛质量或最终性能验收。

## 13. 算子 sub-agent 的职责与严格性能验收

### 13.1 子任务边界

可独立分派的任务包括：重写CR1/CR2 compressor纯AIV算子；W4A16 288尾块/BF16累积kernel；Engram hash/gate算子；CSA2 candidate/indexer性能缺陷。接口未稳定时先做数值/设计核查，接口冻结后再编码，不让多个agent同时改同一binding、CMake或runner文件。

每个算子子任务开始前必须给出：输入/输出/状态的shape与dtype、alias/mutation与graph契约、独立reference、真实shape矩阵、目标性能与baseline、允许修改的文件、完整构建与测试命令。交回结果必须包含代码、正确性结果、性能原始数据和已知限制。主任务负责共享注册/构建清单、模型集成、8卡测试与性能复核；sub-agent报告不能替代集成验收。

### 13.2 性能门槛的冻结方式

当前尚无V4.1在910B的可信性能基线，不能编造绝对microseconds或tokens/s目标。P1先实测每类shape的正确参考实现和可用CANN/CATLASS方案，记录结果后冻结数值目标，再验收优化实现。不能在看到最终实现成绩后放宽门槛使其过关。

默认性能验收规则（有更具体业务SLO时以更严格者为准）：

1. 在专用空闲设备、固定功耗/频率条件尽可能一致时，每case先warmup，再分至少5轮测量；每轮执行足够多次使计时稳定，同时保存中位数、P95和轮间波动。kernel使用NPU event或profiler测量，端到端使用包含必要同步的wall clock，明确测量范围。
2. 针对已有正确可用路径的替换，核心真实shape中位延迟不得回退超过3%，P95不得回退超过5%；若测量噪声已接近该范围，先扩大样本/隔离噪声，不能宣称通过。任何必要的正确性修复造成回退，先作为单独baseline记录，随后达标才标记优化完成。
3. 声称“优化”的kernel，目标负载加权几何平均speedup至少1.10×，且满足上述关键case回退约束。权重由prefill/decode实际流量或事先约定测试矩阵确定，不事后挑case。达不到收益的复杂实现不进入默认dispatch。
4. 新compressor不能以错误旧kernel作为正确性或唯一性能baseline。对比正确的独立projection + 简单CR1/2 reference路径，分别验收GEMM、向量压缩、state操作、cache store和完整链路。优化不能把GEMM重新揉进compressor。
5. 真实尺寸是必测项：MoE H5120 / I288或2304 / top6；compressor T=1/2/4/8/16/32/64/128/512/2048/4096及CR2各种chunk边界；Engram不同batch、随机/重复IDs、NUMA位置。128维toy或单次耗时不作为性能证据。
6. 内存门槛同样强制：无整表NPU暂存、无全MoE双layout常驻；workspace按声明上界增长；长时间replay的device/host pinned用量不持续增加。性能提升不能依赖超出64GiB预算的临时配置。
7. 优化默认启用前必须回到8卡端到端负载验收；关键workload的TPOT/TTFT P95不得回退超过5%。通信、host lookup和调度成本必须计入，不能用单核kernel提速掩盖整机回退。

在记录了实际数字的性能表未通过前，算子状态为“功能可用，性能未验收”，不标记完成。最慢case、cold访问、空/极不均匀expert分布都纳入报告。

## 14. 全部适配后的 profiling、优化与报告

完成文本、graph、vision、DSpark和计划启用的服务组合后，固定一个全功能正确版本，执行整机profiling。此前的局部microbenchmark不能代替这个阶段。

1. **建立整机baseline**：记录标准自回归与DSpark、eager与graph、prefill/decode、不同context/concurrency、Engram冷/热访问；同时报告精度、TTFT/TPOT/吞吐和HBM/host峰值。
2. **采集代表性msprof trace**：覆盖CPU调度/lookup、H2D/D2H、HCCL、Cube/AIV、kernel间空洞、同步等待、graph replay。解释每一类主要耗时的输入依赖，不只给算子排行榜。
3. **按收益排序优化**：优先解决layer1 Engram等待、NUMA和H2D粒度、MoE低利用率或通信、CSA2多余工作、独立compressor projection/norm瓶颈、graph边界空洞。每个改动都列假设、证据、预期收益和内存/精度成本。
4. **逐项A/B并回归**：一次改一个主要变量，重新执行数值和性能门槛；无收益或回退的方案不进入默认路径。完成后再做组合测试，检查优化相互影响。
5. **输出报告**：建议落盘 `docs/performance/deepseek_v41_910b.md`，原始JSON/CSV及trace路径写入报告；若trace过大不进git，则给出稳定保存位置、checksum和重采命令。

报告至少包含：硬件/软件/源码SHA、模型与量化manifest、启动与benchmark命令、完整workload矩阵、正确性与量化质量、优化前后TTFT/TPOT(P50/P95/P99)/吞吐、host/HBM峰值、热点时间占比与timeline、每项优化的A/B结果、性能门槛通过情况、未解决瓶颈和不支持组合。最终结论必须能从保存的原始数据复算。
