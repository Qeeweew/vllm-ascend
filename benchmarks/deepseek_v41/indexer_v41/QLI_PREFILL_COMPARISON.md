# QLI prefill 对照：r17 为什么在 4K / 32K 退化

状态：对照已测，优化未验收；暂停 ring 实现，先明确根因。NPU0 独占，
同一个 r17 隔离 OPP、同一个 r3 Torch binding、H32/D128 INT8，两个分支使用
完全相同的 query、paged K、scale、weights、候选和 causal 长度。完整 selector
均包含相同的升序排序和 padding 后处理；每个输出先通过独立 CPU oracle。

## 三个实现不是同一个对象

1. **本文实测 legacy** 是 vllm-ascend 的 `QLIV2Preload` /
   `quant_lightning_indexer_v2_service_cube_arch22.h`。给 v3 传显式零
   `output_idx_offset` 强制它走普通分支，数学输出不变。
2. **r17 fused** 在同一 QuantLightningIndexerV2 算子内由 `candidateFused`
   分支进入 candidate Cube/Vector 实现。其 Cube 借用了 qli_opt 的三 Key
   buffer / paired WS 流水，但调度改为一个 query M32。
3. **原 qli_opt** 是 `sources/ops-transformer-qli-opt` 的
   `2ed905fcbb517c8d8f149774382df694466f9fbd`，主要优化文件为
   `attention/quant_lightning_indexer/op_kernel/arch22/quant_lightning_indexer_service_cube.h`。
   它没有本次 2048 block candidate consumer 输入/去重/无效页契约，不能把它的
   历史 dense H64 数字叫作当前 H32 candidate 的同输入实测。这里完成了源码
   数据流对照，没有宣称运行了原 qli_opt 的等价 candidate benchmark。

两个实测分支在 profiler 中可共用同一外层 kernel 名，因为它们来自同一 fat
binary；是 tiling flag / 显式 offset 决定内部实现。r17 的 source 和 binary
SHA、显式 fused dispatch 日志、profile 路径均保留在 artifacts。

## 完整 selector 实测

单位 µs；speedup=legacy median / fused median；内存是增量峰值 allocated MiB。

| T | Context | Fused median / P95 | Legacy median / P95 | Speedup | Fused / legacy MiB |
|---|---:|---:|---:|---:|---:|
| 128 | 4097 | 807.44 / 816.65 | 353.29 / 354.43 | 0.438x | 36.50 / 98.77 |
| 128 | 32771 | 1333.32 / 1348.15 | 1072.55 / 1086.83 | 0.804x | 36.50 / 99.02 |
| 128 | 131075 | 1397.55 / 1401.55 | 2286.09 / 2343.20 | 1.636x | 36.50 / 98.77 |
| 512 | 4097 | 2424.60 / 2437.18 | 878.36 / 879.33 | 0.362x | 95.00 / 99.52 |
| 512 | 32771 | 4025.90 / 4041.79 | 3382.32 / 3390.73 | 0.840x | 95.00 / 99.52 |
| 512 | 131075 | 4407.73 / 4409.51 | 6955.92 / 7102.71 | 1.578x | 96.00 / 100.52 |

32K 的 T128 慢 24.3%，T512 慢 19.0%；4K 明显退化。128K 的 T128/T512
分别约 1.64x/1.58x 更快。不能只选择长 N 结果来宣布 prefill 已完成。

## 源码确认的关键差别：prefill 的 K 复用丢失

legacy kernel 定义 `S1_BASE_SIZE=4`、`S2_BASE_SIZE=2048`，并设置
`mBaseSize=s1BaseSize*gSize`。当前 H32 ⇒ M128，每个 QK L0 tile 是
[M128,N128,K128]，一次 K tile 服务同请求的 4 个 query。

它的 Cube service **没有 candidate 分支**：仍对 dense causal 范围做 QK，
Vector 在 `BuildCandidateMask` 后压低候选外 score，再排序。不同 query 的
candidate 集合不会破坏这份 dense K 复用。候选数不足 topk 的旧实现还存在
候选外填充语义；本文性能用例候选足够，通过相同严格 oracle，不能把这份
legacy 当作所有无效候选边界的正确替代。

r17 每次 M32、N128、K128，只计算一个 query 的候选。候选排序和合并相邻
物理 block 后从 paged cache 读取 K，但是同一 K 不再跨 query 复用。
忽略小的 causal 尾部，GM→L1 的逻辑 K 字节数如下（不是 HBM 实测流量）：

| Context | legacy 每 query（4 query 共用） | fused 每 query实际有效 K | 影响 |
|---|---:|---:|---|
| 4K | 128 KiB | 约 512 KiB | 约 4x K 字节，且 fused 仍做完整16K槽 QK |
| 32K | 1 MiB | 约 2 MiB | QK算术减半，K字节反而约2x |
| 128K | 4 MiB | 约 2 MiB | 候选裁剪开始同时节省算术与K字节 |

精确到测试候选/causal长度的逻辑估算保存在 `work-ledger.json`；其中约每7行
有一行候选缺一部分，故精确字节比与名义值稍有差别。legacy 按 page32 一次
搬32行；fused 以候选block8为基准，只有相邻且同页的块才合并搬运，MTE描述符
数也明显更多。候选 scale 的 prep 同样有大量16B短搬运。

QK 只按 K 字节计的算术强度，r17 M32 为64 ops/B；legacy H32 四 query
共用为256 ops/B；原 qli_opt H64四 query共用为512 ops/B。H64→H32降低了
每次 K 搬运对应的有效计算，但无法解释本次两路**同为H32**时丢失的额外4x
query复用；不能把这两项混成一个因素。

## 全核 profile 时间账本

下表是20个 AIC / 40个 AIV 的均值与[min,max]，不是core0代表全卡。
MTE、FixPipe、Cube、Vector可以重叠，不能把这些列相加成 wall time。
READY id6 是 r17 启动计算之前等待全query prep 的独立前置等待。

### T128 / 32K

Kernel wall: fused 1241.24 µs，legacy 974.56 µs。

| 计数 | fused mean [min,max] | legacy mean [min,max] |
|---|---:|---:|
| Cube active µs | 62.15 [57.73,68.41] | 93.13 [58.09,116.65] |
| Cube ratio % | 5.01 [4.65,5.51] | 9.56 [5.96,11.98] |
| AIC MTE2 µs | 556.46 [523.68,613.81] | 147.27 [86.16,208.87] |
| FixPipe µs | 546.76 [515.49,605.20] | 138.36 [84.51,175.56] |
| FixPipe active GB/s | 14.28 [13.87,15.24] | 112.37 [107.35,115.74] |
| READY wait µs | 437.81 [436.73,441.32] | 不同flag协议 |
| ACK wait µs | 78.10 [60.42,99.51] | 不同flag协议 |
| Vector active µs | 139.66 [123.26,182.03] | 676.18 [387.45,873.39] |
| AIV MTE2 µs | 315.70 [268.55,403.03] | 69.62 [42.46,88.63] |
| AIV scalar µs | 528.20 [471.99,664.19] | 598.71 [339.69,778.76] |

### T512 / 128K

Kernel wall: fused 4400.18 µs，legacy 6906.26 µs。

| 计数 | fused mean [min,max] | legacy mean [min,max] |
|---|---:|---:|
| Cube active µs | 248.14 [241.60,252.87] | 1490.20 [1394.51,1632.73] |
| Cube ratio % | 5.64 [5.49,5.75] | 21.58 [20.19,23.64] |
| AIC MTE2 µs | 2755.73 [2675.51,2825.08] | 3364.91 [3128.95,3634.17] |
| FixPipe µs | 2726.19 [2639.95,2800.21] | 2621.53 [2431.73,2873.80] |
| FixPipe active GB/s | 11.46 [11.14,12.02] | 95.10 [91.96,99.61] |
| READY wait µs | 1330.80 [1329.46,1334.50] | 不同flag协议 |
| ACK wait µs | 124.24 [110.02,151.57] | 不同flag协议 |
| Vector active µs | 558.18 [519.47,578.22] | 5251.51 [4907.35,5746.18] |
| AIV MTE2 µs | 1171.82 [1084.03,1217.00] | 1610.17 [1491.59,1735.73] |
| AIV scalar µs | 2100.53 [1964.96,2160.24] | 3616.21 [3364.33,3955.85] |

T128/32K：fused 的 Cube active 均值已经从93.13降到62.15 µs，说明减少算术
确实生效；但 AIC MTE2 从147.27增至556.46 µs，FixPipe从138.36增至546.76 µs，
并新增437.81 µs全query READY前置等待。Vector排序计算从676.18降至139.66 µs，
不足以抵消前述开销。这是比单个“Cube比例低”更完整的退化解释。

FixPipe active带宽从约112.37跌到14.28 GB/s。源码确定的新QK tile M32、
Score L1仍按M128 stride分配，以及更碎的QK/FixPipe调用，是需要独立验证的
具体嫌疑；这些 counter 本身不能把每微秒唯一归因给padding或某个bank冲突。
不能把“拷贝字节更少”直接推成“FixPipe应该更快”。

T512/128K仍有约1330.80 µs的READY等待；但legacy对dense128K做约8倍候选
QK，Vector处理dense score也更多，所以fused总体更快。此时20核平均Cube比例
仍只有5.64%，不意味着省掉的算术应该补成无用矩阵乘来提升比例。

## 原 qli_opt 流水对应到当前实现

原 optimized Cube 保留 Query/Weight双缓冲、Key三缓冲、Score L1双stage、
每stage两个N128 tile，并将WS合并为N256。r17借用了这份流程，但把实际M
固定成32；当前legacy是Key双缓冲、逐QK tile/WS交替的另一份service。
因此“拷了optimized流水”并不能补回丢失的四query K复用。

原文档的典型 H64/M256 拆成两个M128 QK tile，K在这两个M tile及四query
之间复用；H32下原优化结构对应两个M64 tile。r17只有一个M32实际query。
原报告中的dense H64 32K prefill、topk2048、约50% Cube，与这里H32、候选
16K、topk512、T128/512不是同一工作量；没有可支持直接倍率比较的等价实测。

## 有用工作、执行工作和下一步

`work-ledger.json` 分开记录有效候选QK FLOPs、按已确认tile几何推导的发射
QK/WS FLOPs、逻辑K字节，明确它不是硬件retired-instruction计数。4K下r17
执行QK槽位约为有效位置的4.3–4.5倍；32K/128K约1.05倍。WS受Cube最小M16
限制，只有一行有用，不能把16行输出都当有效算法工作。

相邻4query candidate union 的实际测试集合估算：4K额外计算约5%，32K约30%，
128K约49%。这是合成候选的相关性，不能替代真实模型source候选分布。它说明
跨query复用必须量化union开销，不能直接把不同candidate集合拼在一起。

后续先保持这些对照不变。短N/高T可以保留dense共享K的M128，并融合候选mask
和单AIV完整query topk；长N采用candidate路径。全query prep改为bounded ring
的协议作为后续候选方案，尚未实现。N256紧凑H32布局需要独立构建/精度/性能
验证。固定split1/2/4/8完整隔离包已生成；阈值仍待同shape实测，不由启发式推定。

## 可复核材料

- `artifacts/qli-fused/r17/perf-t*-r1.json`：6组完整selector、oracle、内存、raw samples。
- `artifacts/qli-fused/r17/profile-all-core-summary.json`：20+40核所有指标分布。
- 同目录`profile-*/OPPROF*/PipeUtilization.csv`、`OpBasicInfo.csv`：原始counter。
- `artifacts/qli-fused/r17/work-ledger.json`：有效/执行几何工作量、候选union估算。
- `package.json`：source、kernel、tiling library和完整包SHA；所有包隔离安装。

## Candidate contract audit and r19 epilogue experiment

The model wrapper sorts the final sparse **position indices**, but returns the
source **candidate blocks** without sorting them
(`models/deepseek_v4/indexer.py`, `AscendIndexerV41Ops.select_topk`). The source
Vector kernel extracts block IDs from its score-ranked candidate top-k. Thus
candidate block input is not ascending by position. Arbitrary order, duplicate
IDs, invalid pages, negative sentinels and graph padding are also covered by the
existing consumer oracle. Removing all sorting/deduplication would change this
contract; doing so requires an explicit separately validated source/consumer
interface, not an assumption about the current producer.

The first simplification experiment removes **scale** preparation from the
all-query READY dependency. Scale is not needed by the QK/weight Cube pipeline;
it is multiplied into the scalar score in the Vector epilogue. In r19's isolated
source, each top-k owner loads scales directly from the paged cache for its
current 1024-score tile using the already validated physical K offsets. This
removes the 64 KiB/query padded scale GM record and both the GM write/read of that
record. Row workspace becomes 91,168 bytes instead of 156,704 bytes. One
repeat-stride Mul handles the 128 blocks8 of each epilogue tile instead of 128
separate calls. Address preparation, candidate sorting and the existing paired
READY/SCORED/ACK protocol are unchanged for this isolated experiment.

This does **not** yet restore four-query K reuse, merge the remaining short scale
reads, eliminate the all-query address-preparation barrier, or prove a speedup.
Those must not be represented as completed. r19 requires a complete isolated
build, adversarial/ragged/changed-content graph correctness and the same
4K/32K/128K selector and all-core profiler comparison before acceptance.

## r18 invalid-tile trimming result

The full isolated package passed all 52 fused correctness/graph cases.
Short-context all-invalid leading/trailing N128 tiles are skipped while keeping
the two-tile Cube prologue. Median whole-selector time in microseconds:

| T | Context | Fused | Legacy | Legacy / fused |
|---|---:|---:|---:|---:|
| 128 | 131075 | 1472.32 | 2365.84 | 1.607x |
| 128 | 32771 | 1391.98 | 1069.73 | 0.768x |
| 128 | 4097 | 605.33 | 350.05 | 0.578x |
| 512 | 131075 | 4629.24 | 6795.39 | 1.468x |
| 512 | 32771 | 4217.41 | 3393.52 | 0.805x |
| 512 | 4097 | 1724.14 | 878.98 | 0.510x |

This reduces the 4K cost relative to r17, but still fails the legacy latency
gate. At 32K/128K it adds tile-validity preparation/control overhead and has not
shown an improvement over r17. It is not accepted as the final optimization.
The package, source and kernel hashes are in `artifacts/qli-fused/r18/package.json`;
correctness/build logs and raw performance samples are preserved beside it.

The r18 all-core profiler confirms that trimming did not solve the remaining
bottleneck. At T128/4K, wall time is 513.02 us, mean AIC Cube active time is
15.40 us (3.00%), and the initial READY wait is 333.64 us. At T128/32K, wall
time is 1312.32 us, Cube active time 59.16 us (4.51%), initial READY wait
525.79 us, AIC MTE2 553.94 us and FixPipe 535.54 us. These are means over all
20 AICs; raw files include all 40 AIVs too. Counters overlap and must not be
summed. The 32K READY wait increased relative to r17, so the additional per-block
tile-validity work must also be measured or limited to the short-context path.
See `artifacts/qli-fused/r18/profile-all-core-summary.json` for every core's
aggregate range and the adjacent msprof CSVs for the original measurements.
