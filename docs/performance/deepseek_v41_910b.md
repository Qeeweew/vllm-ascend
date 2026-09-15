# DeepSeek V4.1 Flash：910B × 8 阶段验证与性能报告

**状态：未完成全模型验收。这是阶段报告，不是最终整机性能结论。**

截至 2026-09-15，本分支已完成独立算子、host offload 与有限范围
TP8 eager/graph 集成验证。完整 40 层真实权重、两层真实全尺寸 Engram、
长上下文、DSpark 和最终整机 profiling 尚未验收。下面保留通过、失败与
待验证项各自的证据范围；算子加速比不能相乘或外推为整机加速比。

本文仅整理已归档数据，没有新增 NPU 测量。后续运行完成后应更新原始
结果链接与状态，不能把准备脚本或正在运行的任务计为通过。

## 1. 环境与构建基线

| 项目 | 本阶段有效配置 |
| --- | --- |
| 硬件 | 8 × Ascend 910B3，每卡 65536 MiB HBM，aarch64 host |
| CANN / driver | 9.1.0 / 25.5.0 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4；通过 torch-npu 使用设备 |
| Python / Transformers | 工作区 Python 3.12 `.venv` / 5.14.1 |
| vLLM | 本地 editable `sources/vllm`，HEAD `836bb3839ffefcda8283ea7d41671a89e1a613df` |
| vLLM 有效包版本 | `0.1.dev1+g836bb3839.empty` |
| 个人仓库 | `https://github.com/Qeeweew/vllm-ascend.git` |
| 开发分支 | `deepseek-v41-910b-w4a16-engram` |
| 插件基线 | `b49962987e89b850586f1819ce8f85daa85a0f81`；适配提交记录见分支历史 |
| 插件有效包版本 | `0.1.dev5255+gb49962987.d20260915`，工作区 editable |
| 主要集成范围 | TP8 / PP1 / DP1，BF16 activations，group32 W4A16 routed MoE |

系统原有 `/vllm-workspace` 环境未作为本报告有效安装来源。工作区
`.venv` 继承 vendor site-packages；它不是依赖解析完全无冲突的环境。
HTTP app 构造已通过，实际版本和遗留冲突见
[依赖审计](../../benchmarks/deepseek_v41/SERVING_DEPENDENCY_AUDIT.md)。
其中 profiler 可选依赖与 NumPy/OpenCV 元数据冲突仍保留，不能用
HTTP import 成功宣称全部分析工具可用。

新算子通过完整 `pip install -e` 构建安装。早期发现生成目录中的
`.done` 标记会复用旧 kernel，已修复清理流程；后续数值结果应关联到
具体 binary/source fingerprint，不能只引用分支 HEAD。Compressor
r10 指纹见
[r10_binary_manifest.json](../../benchmarks/deepseek_v41/compressor_v41/r10_binary_manifest.json)；
Engram r9 安装对象 SHA256 与日志见
[ENGRAM_GATE_STATUS.md](../../benchmarks/deepseek_v41/ENGRAM_GATE_STATUS.md)。
Candidate 使用已完成的 r12 全量算子构建，当前安装二进制和接线源码见
[r12 manifest](../../benchmarks/deepseek_v41/indexer_v41/r12_candidate_integration_manifest.json)。
这是接线后的快照，不回填此前 benchmark 的 Python 修订；旧版结果不覆盖后续二进制。

## 2. 权重与量化清单：仍不完整

源目录为 `/mnt/models/DeepSeek-V4.1-Flash`，目标为
`/mnt/models/DeepSeek-V4.1-Flash-W4A16-G32`。本次只读检查目标
`conversion_manifest.json` 得到：

| 字段 | 观察值 |
| --- | --- |
| `complete` | `false` |
| 已转换 shard entries | 46 / 48 |
| 已转换文件 bytes 合计 | 331721285976 |
| 缺源文件 | `model-00047-of-00048.safetensors`、`model-00048-of-00048.safetensors` |
| 最终 `config.json` / `model.safetensors.index.json` | 均未发布 |
| format version | 1 |
| Manifest SHA256（本次快照） | `c7c17631cc6b1f88fe7e9ffd02fb139b4c1a3cbdaaede7629b18a4afa9b5f616` |
| Source config SHA256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Source index SHA256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

该 SHA256 标识未完成清单的快照，续转后会变化。每个已转换 entry
记录输出 SHA256、大小与 tensor dtype/shape；原始前 46 片的 header/
大小指纹不等于全 payload 校验。真实全表发布需要完成最后两片 SHA256
校验、续转与全输出 `--verify-only`，并确认 `complete=true`。

转换契约为普通 FP8 反量化到 BF16；原有 BF16 dense 与 FP32 控制参数
保持所需精度。MXFP4 routed experts 转 INT4，沿 K 轴 group32，BF16
signed scale，依据已存 scale 做 nearest-even RTN；checkpoint 使用
`q+8` offset-binary，runtime 转为两个计算路径共用的 signed packing。
全零组 scale 为 `1.1920928955078125e-07`。Engram 输出为 BF16，
原始表 scale block 是每行 32 列，不能套用 dense 的 32 × 32 block。

转换抽样记录在
[conversion_samples.json](../../benchmarks/deepseek_v41/host/conversion_samples.json)，
真实全表后续校验步骤见
[ENGRAM_REAL_TABLE_ACCEPTANCE.md](../../benchmarks/deepseek_v41/ENGRAM_REAL_TABLE_ACCEPTANCE.md)。
转换器已处理 `mtp.*` draft 权重，但这不代表 DSpark loader 或执行通过。

## 3. 算子性能：严格限定到实际测量范围

### W4A16 routed MoE

原生 AscendC 路径复用个人分支优化并修复 I288 尾块、FP32 累加和
同步问题。当前只允许明确 decode 的 B≤4 / E384 / top6 / H5120 /
local-I288 / BF16 / group32 / TP 无 EP 路径，默认开关仍为 false。
Prefill、混合批次及不支持 shape 使用 CANN fallback。

下面是 1000 次 warmup、5 × 300 event 样本的 hot-expert graph 重测；
每 token 使用同一组六个专家。CANN baseline 包括实际 route 初始化、
两次 grouped matmul、clamp/SwiGLU 和 unpermute；不包括 TP collective
及 shared experts。

| B | Native median / P95 µs | CANN median / P95 µs | 最保守轮间速度比 |
| ---: | ---: | ---: | ---: |
| 1 | 98.19 / 111.64 | 316.06 / 331.64 | 2.87× |
| 2 | 128.98 / 148.22 | 332.32 / 339.58 | 2.31× |
| 4 | 240.96 / 250.98 | 340.32 / 351.02 | 1.37× |

初始 hot graph B8 native 为 416.82 / 429.86 µs，CANN 为
352.77 / 359.10 µs，发生回退，因此不能扩展 B≤4 的 threshold。
部分 native round median 仍有 3–12% 波动，未锁频，不证明细小百分比
改进稳定。移除 CANN 可省略的 zero-offset tensors 经逐值验证，可减少
约 101.25 MiB/层，即 40 层 3.955 GiB/rank 常驻存储；这不是完整峰值。

原始样本、基准方法、23 + 1 项 NPU 检查及 150 项 UT 范围见
[W4A16_STATUS.md](../../benchmarks/deepseek_v41/W4A16_STATUS.md)；核心重测为
[w4a16_decode_graph_hot_steady_910b3.json](../../benchmarks/deepseek_v41/w4a16_decode_graph_hot_steady_910b3.json)。

真实前三层捕获的 decode 激活上，native 相对独立 FP32 契约 NRMSE
分别为 `8.3445e-7 / 0.00013825 / 0.00034110`；native 与 CANN
约 `0.00502 / 0.00496 / 0.00517`，主要对应不同 BF16 rounding
边界。原生 FP32 atomic 累加仍有极小重复差异；100 eager + 100 graph
重复的最大 NRMSE 为 `2.669e-5`。不宣称 bit-exact 或全模型质量通过。
参见[真实数值诊断](../../benchmarks/deepseek_v41/W4A16_REAL_NUMERICS.md)。

### 重写 Compressor，矩阵乘独立

Compressor kernel 不含 GEMM，只做 CR1/CR2 向量处理、状态环与 RMSNorm。
CR1 接收 BF16 `[T,512]`，CR2 接收真正 FP32 `[T,1024]` 的 `[kv,score]`。
CR2 保留 pooling 后 BF16 roundtrip，再做 FP32 normalization；RoPE、
indexer projection 与 cache insertion 独立。每 AIV 使用 22 KiB UB，
没有算法级 GM workspace 或跨核同步。

接受的是修复 Sqrt+Div 后的 r10 graph 数据：38/38 主矩阵通过，
geomean 相对明确的 batched PyTorch baseline 为 12.9719×，最大
round-median spread 0.7891%；另 12/12 closing-group cases 通过。
门槛为 median≤baseline×1.03、P95≤baseline×1.05、spread<3%，
aggregate speedup≥1.10。每 graph 含 256 calls，5 × 20 event samples。

| 路径 | T | Median / P95 µs | 相对 baseline |
| --- | ---: | ---: | ---: |
| CR1 decode | 1 | 1.483 / 1.488 | 19.25× |
| CR1 prefill | 4096 | 55.663 / 55.677 | 2.40× |
| CR2 closing decode | 1 | 2.358 / 2.362 | 26.44× |
| CR2 closing decode | 128 | 13.247 / 13.252 | 16.50× |
| CR2 prefill | 4096 | 93.286 / 93.557 | 3.42× |

独立真实 BF16 compressor weights + synthetic activations 的 GEMM
验证采用 `torch.mm(..., out_dtype=torch.float32)`：T1 为 10.886 µs，
T4096 为 173.108 µs；相对 FP64 抽样 oracle NRMSE 为
`2.93e-7–3.43e-7`。BF16 output 后再 `.float()` 的错误控制约
`1.64e-3–1.71e-3`，因此不可替代该接口。

原始[主矩阵](../../benchmarks/deepseek_v41/compressor_v41/graph_r10.json)、
[闭组矩阵](../../benchmarks/deepseek_v41/compressor_v41/graph_closed_r10.json)、
[projection](../../benchmarks/deepseek_v41/compressor_v41/projection.json)
及[完整报告](../../benchmarks/deepseek_v41/compressor_v41/report.md)可复核。
历史 eager 虽通过相对延迟比较，38 个 case 中 25 个超过噪声门槛；
**eager 严格性能验收仍未通过**，不能以 graph 数据代替。

### Engram gate 与全尺寸 host offload

Gate r9 的 17 项 correctness 与 8 个 graph 性能 case 全通过。
只测 post-wkv normalization、gate 与 residual；GEMM、host gather、
H2D 和 collective 不在计时内。T1 median/P95 为 3.551/3.566 µs，
T1024 为 231.874/231.896 µs，相对 device composition baseline
29.86×/6.42×。所有 NRMSE <`2e-4`，masked rows 精确保持；gate
要求不同 T 桶 25/40/250 µs 上限及≥2×加速。采用 graph 内 32 calls、
每 event 10 replays、50 samples，表示摊销设备时间。

[engram_gate_910b.json](../../benchmarks/deepseek_v41/engram_gate_910b.json)
保留原始值。早期 basic `Rsqrt(1)=0.998046875` 引起的精度失败未被
放宽 tolerance，最终改为 Sqrt + 精确 division。

完整 host **容量**独立通过：8 ranks、16 owners 同时注册并驻留
393227699200 B，即 **366.221833 GiB** BF16 production-shaped
合成表。400 个采样页与 16 个全 VMA NUMA 记录匹配；144 行首/中/尾
DMA、160 次变化 hash/DEAD/padding 的 staged offload + graph 精确通过。
CPU gather 在 graph 外，模型 graph 只读固定 device row buffers。

每 rank 约 45.78 GiB host；该诊断 torch reserved 为 6 MiB/rank，
不包括所有 driver/context 内存。每张约 22.89 GiB 表注册 wall median
9.688 s，unregister median 0.943 s，是初始化/清理成本，不是 token
延迟。16 次 checked unregister、8 子进程与 controller exit0 均通过。
原始数据见
[engram_full_capacity_910b.json](../../benchmarks/deepseek_v41/engram_full_capacity_910b.json)，
范围与加载峰值缺项见
[ENGRAM_FULL_CAPACITY_RESULT.md](../../benchmarks/deepseek_v41/ENGRAM_FULL_CAPACITY_RESULT.md)。
**该测试没有读取真实 Engram checkpoint，不能据此关闭真实全表验收。**

### Candidate selector

可选 B1 consumer 使用独立 AscendC gather、BF16 BMM→FP32、AscendC
score，后接 topk/remap/sort。33 项 correctness/dispatch/变化输入
graph 检查通过，默认开关仍关闭。三组 compressed context 的最终
whole-selector 数据如下，包含 candidate preprocessing：

| Context | Candidate median / P95 µs | Live dense median / P95 µs |
| ---: | ---: | ---: |
| 4097，独立重测 | 69.472 / 69.789 | 91.120 / 93.050 |
| 32771 | 81.312 / 81.710 | 128.675 / 130.730 |
| 131075 | 81.094 / 81.661 | 199.698 / 206.855 |

Frozen/live latency 和 spread≤3% 门槛均有通过记录。4097 初测 dense
spread=3.2898% **失败**，其后按原门槛独立重测通过，不能删除初测。
先前仅把 INT32 转 FP32 的改动无性能收益；有效收益来自以合法 position
判断 validity，避免 score 与负无穷比较。展开 profiler 的进一步归因
尚未运行，不能把整个收益归给 INT32 `where`。

[实验报告与原始结果](../../benchmarks/deepseek_v41/indexer_v41/CANDIDATE_EXPERIMENT.md)
记录 3 × 12 samples、candidate/dense graph unroll 64/4 与逐 stage
计时。Candidate 的后续 TP8 **合成模型功能验证已通过**：40 层/E8，
prompt 1152/1160，prefix cache on、NUMA 与 strict HCCL，每 rank
candidate capture4/eager4，日志确认实际 ACL graph replay。12 个输出
token 与 selected logprob 和历史 `graph_40_long.json` 逐位相同，
同实例重复也逐位相同；8 rank owners 释放，EngineCore exit0。
原始结果为
[graph_40_candidate_production.json](../../benchmarks/deepseek_v41/runner_tp8/graph_40_candidate_production.json)，
日志 `/tmp/v41-runner-40-candidate-production.log`。这不是 40 层真实
权重、真实全表或性能 gate；默认仍 opt-in false。B8/B32、原 selector
的其他失败 gate、长上下文质量和整机速度仍独立。

## 4. TP8 与视觉：真实权重范围、失败和清理

已运行的真实语言 fixture 是 **3 层、384 experts、一个 4096-row
合成 Engram**，通过标准 safetensors loader 装载。另有 40 层合成
fixture 验证结构，但它不等于 40 层真实模型。

最初 CANN 同输入重复 selected-logprob 差异定位到首层 `wo_b` 后
BF16 HCCL reduction：8 rank 的 local matmul 逐值相同，reduction
后有 93357/163840 元素不同、NRMSE=0.004757636。设置
`HCCL_DETERMINISTIC=strict` 后该追踪样本重复差异为零；strict BF16
仍不等价于 FP32 reduction。原始全 rank 对照和范围见
[ATTENTION_REPEAT_NUMERICS.md](../../benchmarks/deepseek_v41/ATTENTION_REPEAT_NUMERICS.md)。
这项定位不是 native MoE 精度修复或全模型质量验收。

视觉组件使用完整 **32 层真实权重 tower + aligner**。真实 `hato.jpg`
512×340 缩略图经发布 processor 得到 1536 patches；相对发布 CPU
reference，tower/aligner NRMSE 为 **0.0215005/0.0140770**，满足
预定 0.03 gate。此前 1024-patch 合成渐变的 NRMSE 为
**0.1166869/0.0693720，仍失败**；照片通过不覆盖该 stress failure。
参见[VISION_COMPONENTS_REPORT.md](../../benchmarks/deepseek_v41/VISION_COMPONENTS_REPORT.md)。

最大声明单图 shape 另做容量测试：9189 patches、完整 1024-token span，
real tower/aligner finite 与 shape 全通过；peak allocated
**1349122560 B（约 1.256 GiB）**，低于事先 4 GiB component budget。
3 个 steady wall 样本为 **232.126/233.057/233.268 ms**，cold
1351.613 ms。此时间不含 CPU resize、H2D、weight load，不是 TTFT，
3 个样本也不能估计 P95。见
[vision_capacity_npu.json](../../benchmarks/deepseek_v41/vision_capacity_npu.json)
及[容量报告](../../benchmarks/deepseek_v41/VISION_CAPACITY_REPORT.md)。

生产注册入口 MM graph 两项有限范围集成已通过：

| 场景 | 可复核结果 | 原始记录 |
| --- | --- | --- |
| 图片 + literal image-ID 文本 | 189-token 图片 span + 2 text；每 rank 1 graph / 6 replay；native capture3；NUMA row buffers 稳定；checked Engram release 和 engine exit0 | [mm_production_numa_graph.json](../../benchmarks/deepseek_v41/mm_production_numa_graph.json) |
| image limit0 | 无 vision 参数/encoder 分配；raw IDs 文本执行；每 rank 6 replay；checked release 和 engine exit0 | [mm_production_limit0_graph.json](../../benchmarks/deepseek_v41/mm_production_limit0_graph.json) |

上表仍用真实三层语言权重与小合成表。自然照片的 eager/graph token
一致也不建立图像理解质量、并发 MM 或真实全模型峰值内存。

HTTP r1 的 `/health`、`/v1/models`、text completion、SSE 与 image
chat 请求检查通过，但总体 **`failed_cleanup`**：API parent exit0
且无存活子进程，内部 process manager 仍强杀了 EngineCore，并产生
8 semaphore / 10 shared_memory leak warnings。不能计为 HTTP 完整验收。
r2显式shutdown-timeout30已验证8worker正常退出、无强杀/资源泄漏；关闭后的output-handler EngineDeadError时序保留记录。r1见
[http_serving_graph_r1.json](../../benchmarks/deepseek_v41/http_serving_graph_r1.json)
及其 `server_log`，控制日志 `/tmp/v41-http-serving-r1.log`。

三层HTTP graph已采集并离线导出8份真实timeline；每rank含6次graph执行、18次原生W4A16、8次Engram gate和8次compressor。daemon内自动导出失败保留，官方offline analyse成功，无需重跑设备。重复图片命中缓存，本轮不含视觉塔计算；原始与导出文件已保存到工作区artifacts（900文件、285832345字节），逐文件SHA256与全rank统计可复算。详见[HTTP profiling结果](../../benchmarks/deepseek_v41/HTTP_PROFILE_RESULT.md)。HCCL等待在该fixture中突出，但不据此宣称网络或整模瓶颈。

## 5. 待完成验收与 profiling

| 项目 | 本阶段状态与下一份必要证据 |
| --- | --- |
| 源 Engram 47/48 下载 | pending；恢复目录 `.v41-recovery-3bd368ab0f3d` 与 `/tmp/v41-engram-recovery.log` 保留动态进度，完整 SHA256 后才发布 |
| 最终转换 | pending；48 verified shards、`complete=true`、最终 config/index |
| 真实全表 TP8 | pending；直接流式填 owned head ranges、真实行 oracle、16 owner placement、RSS/HBM peaks、checked cleanup |
| 40 层真实推理 | pending；完整权重质量、eager/graph 一致性、长 prompt/多请求、真实 HBM 与 host 容量 |
| HTTP 清理 | r2无强杀/泄漏且8worker正常退出；保留关闭后output-handler时序日志，见HTTP报告 |
| Candidate TP8 | 40 层/E8 合成 graph 功能通过；真实 E384 全模型与整机性能仍 pending |
| DSpark | aux、draft/loader、K5非因果attention与placeholder组件已有实现和组件测试；准入仍关闭，registry、真实权重整体验证及rollback待完成，见 [DSpark audit](../../benchmarks/deepseek_v41/DSPARK_INTEGRATION_AUDIT.md) |
| 整机 timeline profile | 三层fixture已完成8rank采集及离线导出；40层真实整机仍未采集，不推导最终性能结论 |

完整装载仍需确保八卡同时有足够剩余 HBM。此前 NPU7 存在约 34 GiB
外部占用；小 fixture 成功不证明该占用下能放入真实 40 层。该数值是
历史资源阻断记录，下一次全模型启动前需重新只读检查，不能当作当前
实时占用或擅自终止外部任务。当前没有全模型可用性的内存证据。

`msprof --help`、Torch-NPU CPU/NPU profiler 与 timeline/kernel/memory
parser imports 已通过；后续三层HTTP真实采集和离线解析亦通过。
`ms_service_profiler` CLI 因缺 `msguard` 仍失败；这不阻断已可导入的
core Torch-NPU/CANN 路径。工具能力和依赖边界见
[PROFILING_TOOL_READINESS.md](../../benchmarks/deepseek_v41/PROFILING_TOOL_READINESS.md)。

完成实际功能准入后，先采集一份小而非空的真实 trace并成功导出，再
对已验收的 prefill/decode shapes 采样。至少分开观察 CPU Engram
hash/gather、D2H/H2D、stream wait、独立 compressor GEMM/vector、
W4A16/shared experts、HCCL 和 graph replay；记录 rank/step 覆盖和
采集开销，并与无 profiler 的 matched workload 对照。

最终报告需要真正的 TTFT、median/P95 TPOT、delivered tokens/s、
请求长度/并发、peak HBM/pinned host bytes、权重与 binary manifests、
实际 trace 路径和正确性结果。**这些整机性能字段目前均无验收数据，
本报告不填估算值，也不宣称适配完成。**
