# 已撤回：连续 K workspace 方案（历史证据）

**2026-09-16：按用户要求撤回，不再保留可调用的连续 K 实现。**
r23–r29 数据仅用于记录失败实验，不代表当前 mode4，也不用于推导直接分页路径性能。
当前 mode4 只允许 Cube 从原分页 cache 直接搬 K 到 L1；设计见
[PAGED_UNIQUE_DESIGN.md](PAGED_UNIQUE_DESIGN.md)。以下内容均为撤回前的历史记录。

2026-09-16：**正确性通过，性能验收未通过，不默认启用。** r23 完整隔离包
无需旧 vendor 补全即可运行。mode4 在 128K 明显加速，但 4K 退化，32K 对
同输入 dense 路径只有 1.13–1.16x，低于预定的主要 prefill 至少 1.2x 门槛。

## 本轮范围

这是 CR1 consumer 的完整 selector 时间，包含 gather、两次 Cube matmul、
scale/mask、完整 query top-k 及相同 Python 后处理。候选来自实际 source
kernel，但 source 的执行时间不计入 consumer。Q/K/weights 是固定种子的
合成张量，**不是模型 checkpoint activation trace**；不代表整机吞吐。

每个 shape 的三个实现共用同一份输入。`native` 是通过显式零 offset 选择的
原 dense 路径；`paged_candidate` 是 mode2 分页候选融合路径；`fused` 是 mode4
连续 K 双槽路径。这里的 dense 对照来自当前 vllm-ascend 实现，不能称为用户
提供的 qli_opt 分支在本机的直接实测。

图内每次展开4次调用，轮换测试顺序，3轮各12个样本；保留全部原始样本。
每个实现先对独立 CPU oracle 验证，再做计时。设备运行前后均记录进程状态。
六个单请求 prefill shape 不替代多 batch 性能测试；多 batch 目前仅有正确性
证据。mode4 的全核 msprof 对照正在归档，见后续更新；不能用 core0 替代全部核。

## 六场景性能

单位 µs，三个实现均列 median / P95；加速比大于1表示 mode4 更快。

| Query tokens | Context | mode4 | dense | paged candidate | 对 dense 加速 | 对 paged 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 4097 | 533.3 / 536.3 | 360.3 / 367.4 | 640.5 / 648.7 | 0.676x | 1.201x |
| 128 | 32771 | 959.6 / 961.1 | 1084.0 / 1086.5 | 1386.5 / 1401.8 | 1.130x | 1.445x |
| 128 | 131075 | 981.4 / 986.3 | 2344.6 / 2399.1 | 1444.0 / 1450.2 | 2.389x | 1.471x |
| 512 | 4097 | 1501.7 / 1514.5 | 913.6 / 918.2 | 1645.4 / 1655.4 | 0.608x | 1.096x |
| 512 | 32771 | 2964.0 / 2968.5 | 3440.4 / 3443.6 | 4196.8 / 4212.2 | 1.161x | 1.416x |
| 512 | 131075 | 2987.1 / 2990.8 | 6906.2 / 7124.8 | 4625.8 / 4638.4 | 2.312x | 1.549x |

4K 的 mode4 median 比 dense 增加约48%（T128）和64%（T512），P95 同样退化。
不能用 128K 的收益覆盖短上下文退化。32K 对 paged 候选路径虽提升约1.42–1.45x，
仍未达到对更快 dense 对照的冻结门槛。

固定用户 workspace 为 `2 * 20 * (157184 + 16384 * 128)` = 90173440 bytes，
即85.996 MiB，不随 T 增长。实测增量 peak allocated 包括 CANN workspace
和其他临时输出，T128 约102.50 MiB、T512 约104.99–105.00 MiB；不可将这个数
描述为仅 K workspace。六场景完整内存数值保存在 JSON。当前 mode4 比 paged
T128 的35.38 MiB更高；双槽有界不等于在所有 T 上用量最小。

## 正确性与构建证据

r23 自包含 OPP、r4 Torch extension 下，`test_indexer_v41_contiguous.py`
**15 passed in 33.17s**。包括 T1/2/4/32/64/128/512 prefill，B1/8/32/64
多请求 decode，负分数、不足512、坏页、空请求与 padding，真实 source
唯一性到 Python trusted consumer，以及 T128 图的16次内容变化 replay。
图 replay 改变 Q/K/weights/scales、长度、分页、候选，跨越多次双槽复用。
六组性能输入的三个实现另有全部 oracle 通过的记录。

完整构建使用 `build.sh --pkg --soc=ascend910b`，选中 QLI 及其 metadata，
确认新 kernel、AICPU library 和 JSON 均安装到隔离 vendor。r22 缺 AICPU
产物，曾依赖 r17 fallback；它不是本轮自包含包。历史 r22 benchmark 的文本
workspace formula 错误，实测 allocation 数有效；此处仅引用已修正的 r23。

证据目录为 workspace 下 `artifacts/qli-fused/r23/`：

- `package.json`：构建命令、source/kernel/package/AICPU SHA256。
- `perf-t{128,512}-n{4097,32771,131075}-source-r1.json`：原始样本、逐实现
  correctness、库路径和指纹、allocated/reserved 内存。
- `candidates-t*-source-r1.pt`：source 生成的候选，供同输入 msprof 重放。
- `matrix-summary-r1.json`：上述结果与 fixture 的 SHA256、加速比和 CPU 候选统计。
- `v41-qli-contiguous-r23-full.log`、`v41-qli-contiguous-r23-matrix.log`、
  `v41-qli-fused-build-r23.log`：保留的测试、性能 driver 和完整构建日志。
- `device-before.txt` / `device-after.txt`：设备进程状态。

包 SHA256：`28a4a1464e50022dbb3c1f08721337083e537df0b2056bfdd3b9e7feb5c994d7`。
Kernel SHA256：`d6e98451a1d0e241434b085c76dd1570cbd0aac4fa926417a0dc27314ee1a5fa`。
Torch extension SHA256：`7345dfa293a9ddbd00efc4264faecde0e14d4faf39d1e2992495c3f974160648`。

## 可以从源码和输入确认的浪费

r23 gather 固定遍历32个 chunk，每个64个 block8，始终写64KiB K 到 GM。
4K source fixture 中每个 query 仅8或9个非空 chunk，约75%的 chunk 全无效，
但目前仍写完整2MiB K/query。T128 的全无效 chunk 对应可避免的 K 写量
201261056 bytes，T512 为805240832 bytes；这只是 GM 指令写量估计，不能
直接等同于 HBM 流量或预期延迟收益。实际 L2 大小192MiB，必须看缓存计数器。

候选在4K中是有效前缀，T128 每query497–513个有效 block，T512 为449–513。
32K/128K均为2048个有效 block，所有32个 chunk 非空。因此跳过全无效 chunk
写 K 只直接针对4K；它不能解释或解决32K的全部性能差距。

Cube 已按最后有效位置裁掉尾部 N128 tile，至少保留启动流水所需256个位置。
但仍继承旧 M128 score stride，实际 query heads 为32。紧凑 H32/N256 布局
尚未实施。不能只把 N128 常量改为256：原 L0B 有4个16KiB槽，N256 QK 和
配对 WS 都需要32KiB槽，必须联动调整槽数、地址和事件配对。

## 下一轮实验

1. 使用已保存 source fixture 对 mode4、dense、paged 三路做 `msprof op`
   PipeUtilization，先测 T128/32K、T512/128K 和4K退化场景。汇总全部20个
   AIC、40个AIV的分布；同时检查 group 间负载差异及 flag 等待。
2. 独立实验跳过全无效 chunk 的 K GM写回，保留必要的 position/scale 初始化
   与同步。对内部空洞、空请求、槽复用、图 replay 做回归。不能改变数值合同。
3. 根据 profile 决定 H32紧凑 score/N256 service 实验，保持 head reduction
   在第二次 Cube matmul；Vector 仅 gather 和最终 scale/mask/top-k。
4. 每次改动完整重建、冻结新包、重跑 native/graph/multibatch oracle 和六组
   性能矩阵。随后补 T1024/更大 graph bucket、多 batch 性能与小 T splitN
   1/2/4 的实测。source/CR2 和完整模型仍需单独验收。

后续全核 profiling 与 r24 实验已完成，结果见下文；H32 r26 构建完成，
等待设备窗口进行NPU正确性和完整selector性能测试。

## r23 全核 msprof 对照

9组 `msprof op --aic-metrics=PipeUtilization` 已完成，全部保持1800 MHz，
每组均检查20个 AIC、40个 AIV 的原始行和运行后的独立 CPU oracle。
profile 程序加载六场景 benchmark 冻结的 source candidate fixture，Q/K
使用相同固定种子和 cache strides。fixture SHA记录在日志和汇总中。

表中活动时间单位 µs，列出全部 AIC 的 mean [min,max]；Cube百分比为20核
均值。Profiler wall time 是单 kernel，不包括完整 selector 的其他后处理，
不能与前文 graph event 延迟混合算加速比。不同管线的计数器可重叠，不能相加。

| Shape / 实现 | Kernel wall | Cube % | Cube active | AIC MTE2 | FixPipe |
| --- | --- | --- | --- | --- | --- |
| t128-n32771-contiguous | 859.04 | 7.02 | 60.27 [55.94,66.47] | 115.69 [104.82,128.92] | 120.01 [111.18,134.47] |
| t128-n32771-dense | 987.94 | 9.44 | 93.19 [58.13,116.75] | 208.18 [119.99,266.28] | 176.14 [100.66,224.35] |
| t128-n32771-paged | 1375.08 | 4.50 | 61.92 [57.62,67.84] | 558.57 [518.68,619.90] | 536.09 [495.25,593.55] |
| t128-n4097-contiguous | 436.70 | 3.48 | 15.16 [14.03,16.71] | 36.39 [31.10,52.81] | 39.64 [34.18,55.09] |
| t128-n4097-dense | 258.38 | 4.47 | 11.53 [7.09,14.65] | 30.46 [21.12,40.45] | 22.56 [11.20,30.66] |
| t128-n4097-paged | 625.68 | 2.59 | 16.16 [14.94,17.82] | 93.20 [82.25,103.25] | 74.90 [60.58,85.32] |
| t512-n131075-contiguous | 2871.16 | 8.40 | 241.19 [234.77,246.80] | 461.34 [436.72,479.18] | 480.83 [462.37,495.67] |
| t512-n131075-dense | 8438.22 | 17.67 | 1490.89 [1395.18,1633.50] | 5698.73 [5409.13,5971.65] | 4248.75 [4074.02,4476.02] |
| t512-n131075-paged | 4487.56 | 5.52 | 247.65 [240.68,252.37] | 2748.54 [2666.18,2867.58] | 2673.01 [2590.75,2791.66] |

T128/32K：mode4 相比同fixture paged，将 AIC MTE2 从558.57降至115.69µs，
FixPipe从536.09降至120.01µs；连续 K 确实消除了 Cube 侧分页小搬运的大头。
Cube本身 active60.27µs，平均利用率7.02%，而 READY 等待均值525.66µs。
gather AIV 的 MTE2 为537.27µs、scalar541.35µs；top-k AIV 的vector272.66µs、
scalar493.56µs。现在继续只调Cube tile不能消除这段gather就绪等待。

T128/4K：Cube active仅15.16µs，READY等待318.93µs。gather AIV 的MTE2为
178.67µs、MTE3为110.98µs、scalar288.34µs；top-k AIV 的vector69.14µs、
scalar129.40µs。结合全无效chunk仍写满2MiB的源码证据，先验证减少无效搬运。
不能把gather管线全部时间直接归因于K写回：它同时包含每block的1KiB K读取、
16B scale读取、页表/候选标量计算与各段同步。

T512/128K：Cube均值241.19µs，占8.40%，READY等待2166.17µs；gather MTE2
2139.84µs、scalar2147.99µs。后续需要减少小DMA/标量开销并平衡gather/top-k，
同时保持矩阵乘法和head reduce在Cube，不为提高利用率而增加无用算术。

`profile-all-core-summary-r1.json`保存九组完整20+40核分布、每核原始值、
CSV/summary SHA及有效QK工作量；各`profile-*/OPPROF*/`保留原始CSV。
这里只测了当前vllm-ascend的dense和paged控制；与原qli_opt的对应关系是
[源码对照](QLI_PREFILL_COMPARISON.md)，未将其历史H64数字冒充同shape实测。

## 目标修正与按长度选择路径的边界

按用户最新说明，每个 query 的候选 K 集合不同，当前连续 K workspace
不具备跨 query 复用。**优化目标是完整 selector 延迟、吞吐和 workspace，
Cube利用率仅用于诊断。** 不要求达到dense/H64的利用率，也不通过增加矩阵
乘法工作或前置处理来追高这个比例。多batch/prefill的已冻结性能门槛保持不变。
矩阵乘法与head加权reduce仍在Cube；此说明不改变用户要求的分工。

r23六个同source输入的prefill shape中，连续gather都快于直接paged candidate：
4K约1.10–1.20x、32K约1.42–1.45x、128K约1.47–1.55x。因此目前没有依据在
这六个shape按长度切回paged来提高速度。paged在T128使用更少workspace，
可以作为后续内存受限场景的权衡；当前没有据此新增自动dispatch。小T、多batch
的性能尚未完整测量，不能把prefill结论直接外推。

原dense控制在4K明显更快，它沿dense K顺序计算并在Vector屏蔽不同query的
候选，所以能够共享dense K搬运；这与共享已gather的不同候选K是两回事。
但它不是当前mode4输入合同的通用替代：

- 现有`BuildCandidateMask`只把候选外分数降到有限NEG_HUGE，保留实际位置。
  不足512个合法候选时可能输出候选外位置；mode4要求这些输出为-1。
- dense Cube按完整可见范围读分页K，现有控制并未覆盖mode4对非法物理页的
  严格排除合同。
- `trusted_unique_candidates=True`只承诺唯一有效ID，不承诺短上下文下
  覆盖所有可见block；现有负sentinel/内部空洞测试必须继续成立。

因此不能只根据context长度就把mode4切到该dense控制。若选择保留dense K
复用的短上下文融合实现，需要同时完成候选membership、无效位置与页边界的
正确性处理，并验证完整selector收益；不得为了启用现有路径放宽oracle。
这与是否需要连续gather是两个独立的实现选择，当前尚未完成新的short路径。

r24仅跳过全无效chunk的K GM写回；初步4K结果与r23几乎相同，不能把减少
指令字节数宣称为速度提升。r25进一步裁掉全负sentinel尾部的gather循环，
保持最后整个1024-position epilogue segment初始化和内部空洞位置不变；
完整构建已完成，尚未NPU验证。这是删除现有无用循环，没有增加新的预处理阶段。

## r24 无效 K 写回实验结果

完整r24包 **18 passed in 53.13s**，新增T1024、混合多请求512行prefill（完整
内部空洞chunk）和2048 graph bucket。六场景三个实现均通过独立oracle。
矩阵和worker正常退出，设备无残留进程。r24包SHA：
`69a43ef2d18a62ec3ae11cee2c91d461e12de080b27a7195e5a496edaac61fae`。

| T | Context | r24 median / P95 µs | median相对r23变化 | 对dense加速 |
| --- | --- | --- | --- | --- |
| 128 | 4097 | 533.05 / 1766.24 | -0.04% | 0.689x |
| 128 | 32771 | 956.56 / 957.32 | -0.32% | 1.133x |
| 128 | 131075 | 975.75 / 985.10 | -0.58% | 2.331x |
| 512 | 4097 | 1495.32 / 1509.90 | -0.43% | 0.610x |
| 512 | 32771 | 2966.15 / 2983.29 | +0.07% | 1.162x |
| 512 | 131075 | 2994.49 / 2997.64 | +0.25% | 2.245x |

**无明显完整selector速度收益，性能门槛仍失败。** 即使4K省去约75%的K GM写
指令字节数，延迟仍几乎不变；不能把搬运字节估计直接当作吞吐收益。这个结果
也提示需要检查真正限制流水的搬运/标量发射，而非仅减少已被重叠的MTE3写。
证据保存在`artifacts/qli-fused/r24/`的package、六组perf、matrix-summary及日志。

## 下一轮按 H32 优先级调优

用户进一步明确优先级为H32 Cube分块与搬运。r25仅sentinel尾循环裁剪包已完成
构建但未NPU验证，保留独立实验，不作为H32实验基线。r26从已验证r24出发，
只开启H32紧凑L1几何：query/weight容量缩为32行，Score stride M128→M32，
N128、K128、4个16KiB L0槽与同步事件数均保持原样。其L1布局为query8KiB、
weight2KiB、key48KiB、score64KiB，共122KiB。默认mode2维持原布局。

r26完整后台构建已通过，repository、编译副本和安装Cube源码SHA一致，
尚无NPU正确性或速度结论。包SHA为
`b6b18f8a987672566c596cf837d69555dd060e4888b897c2c36d99e1932c70f5`，
资源布局与源码核对在`artifacts/qli-fused/r26/compact_cube_source_check.json`。后续N256实验再独立调整
QK/WS的L0B占用和L0事件槽，不能简单改N常量：H32/N256的QK与配对WS各需
32KiB，原H64路径的双槽WS保留策略也应分别核算。每次都测完整selector，
不新增候选排序、去重或跨query候选K复用假设。

当前NPU0已交还根代理进行TP8 DSpark graph验收；H32调优在此期间只做CPU
设计与后台构建，待根代理释放设备后继续原冻结门槛的正确性/性能验证。

N256下一步的资源核算（计划，未实现）：

| 项目 | H32/N128 r26 | H32/N256候选 |
| --- | --- | --- |
| INT8 QK的L0B | 128×128=16KiB | 256×128=32KiB |
| 配对WS的L0B | 32×256×2=16KiB | 32×512×2=32KiB |
| L0AB物理槽 | 4×16KiB | 2×32KiB |
| QK L0C / paired WS L0C | 16KiB / 16KiB | 32KiB / 32KiB |
| query / weight / key / score L1 | 8 / 2 / 48 / 64 KiB | 8 / 2 / 96 / 128 KiB |
| Cube最小启动N | 256 | 512 |

N256需要把paired WS的事件保留按实际32KiB单槽计算，不能照搬旧H64代码的
“两槽占用”到新的2×32KiB布局。M_MTE1事件的初始化和drain必须同时改为实际
槽数；三Key、双Score stage及跨核双槽协议保持。只有这些相关布局/搬运变化
作为一个完整N256实验，其余gather、top-k和数值舍入不变。最终是否采用由
同输入完整selector结果决定，不能仅凭tile更大或L1占用更少选择。

## H32 r26紧凑L1实测：拒绝作为当前默认几何

r26的18项native/changed-content graph通过（72.36s）。原六场景以及新增B4/T128、
B8/T512多请求prefill共8组完整selector均完成，每组mode4/dense/paged三个
实现先通过CPU oracle。单位µs，列median/P95：

| Shape | r26 mode4 | 相对r24 median变化 | 对dense加速 | 对paged加速 |
| --- | --- | --- | --- | --- |
| b4-t128-n32771 | 1046.34 / 1048.52 | 未测r24同多请求shape | 1.483x | 1.345x |
| b8-t512-n32771 | 3275.55 / 3278.79 | 未测r24同多请求shape | 1.052x | 1.284x |
| t128-n131075 | 1038.65 / 1048.03 | +6.45% | 2.173x | 1.388x |
| t128-n32771 | 1020.25 / 1022.26 | +6.66% | 1.062x | 1.358x |
| t128-n4097 | 524.55 / 528.82 | -1.59% | 0.702x | 1.167x |
| t512-n131075 | 3250.92 / 3253.31 | +8.56% | 2.065x | 1.426x |
| t512-n32771 | 3227.83 / 3230.53 | +8.82% | 1.069x | 1.302x |
| t512-n4097 | 1577.71 / 1594.41 | +5.51% | 0.582x | 1.089x |

Score stride M128→M32及query/weight L1容量压紧虽令L1由384降到122KiB，
32K/128K的完整selector反而慢约6–9%，不采用这项单独改动。多请求B4/T128
对dense有收益，但B8/T512仅约1.05x，不能用前者覆盖后者未达门槛。

r26同r23 source fixture的全核profile已完成，见下表；不能仅凭计数器就断言
是padding、bank冲突或某条DMA指令导致退化。`artifacts/qli-fused/r26/matrix-summary-r1.json`
保存完整汇总、原始perf指纹及fixture信息；全部原始samples与native日志均保留。

r27 H32/N256完整包已构建，包SHA
`3864ab43b5a9ca0e99f2316acfbc4a9e9df443a7f1646327f16aec31489a4938`，
query/weight容量及score M32与r26相同，N256/L0槽/WS事件保留按前述核算整体
修改。仓库、编译副本、安装Cube源码一致。其后完成的NPU正确性、完整selector
与全核profile结果见下节；r26退化本身不是采用更宽N的依据。

### r26同fixture全核profile

两个版本使用同一r23冻结candidate fixture，运行后CPU oracle通过，20+40核
原始CSV和SHA保留。以下为所有相应角色核的mean，时间µs，计数器可重叠。

| 指标 | r23 原L1布局 | r26 H32紧凑L1 |
| --- | --- | --- |
| Kernel wall | 859.04 | 913.64 |
| AIC Cube active | 60.27 | 63.09 |
| AIC MTE1 | 128.41 | 145.06 |
| AIC MTE2 | 115.69 | 116.52 |
| AIC FixPipe | 120.01 | 117.58 |
| AIC READY等待 | 525.66 | 568.80 |
| gather AIV MTE2 | 537.27 | 583.03 |
| gather AIV scalar | 541.35 | 567.45 |
| top-k AIV vector | 272.66 | 272.72 |

压紧Score stride没有产生明显FixPipe收益，MTE1时间增长；同时未修改的gather
也出现MTE2和等待变化。因此不能把全部退化简单归因于某个stride或bank冲突。
本轮实际收益为负，拒绝单独采用紧凑布局；保留它只用于与N256配套的独立实验。
完整对照在`artifacts/qli-fused/r26/profile-comparison-r23-r26.json`。

## H32 r27 / N256实测

r27的18项native/graph通过（53.36s），8shape全部完成，三路oracle均通过。
单位µs，列median/P95。负变化表示更快：

| Shape | N256 mode4 | 相对紧凑N128 r26 | 相对原布局r24 | 对dense加速 |
| --- | --- | --- | --- | --- |
| b4-t128-n32771 | 985.83 / 990.59 | -5.78% | 未测 | 1.561x |
| b8-t512-n32771 | 3064.23 / 3065.56 | -6.45% | 未测 | 1.122x |
| t128-n131075 | 989.93 / 990.80 | -4.69% | +1.45% | 2.450x |
| t128-n32771 | 977.38 / 989.45 | -4.20% | +2.18% | 1.116x |
| t128-n4097 | 527.29 / 529.18 | +0.52% | -1.08% | 0.697x |
| t512-n131075 | 3055.54 / 3061.15 | -6.01% | +2.04% | 2.244x |
| t512-n32771 | 3025.96 / 3030.22 | -6.25% | +2.02% | 1.140x |
| t512-n4097 | 1519.76 / 1526.48 | -3.67% | +1.63% | 0.601x |

N256较紧凑N128恢复了多数性能，但与原L1布局r24相比，32K/128K仍略慢。
4K仍输给dense，32K仍未达1.2x门槛，不能默认采用。全核profile已完成并复采，
完整样本和指纹在`artifacts/qli-fused/r27/`。

N512候选r28已完成完整构建：QK M32/N512/K128，paired WS M16/N1024/K32，
L0AB单64KiB槽，L1 query/weight/key/score为8/2/192/256KiB，总458KiB。
它减少tile发射次数，但失去L0双缓冲、最小启动N增加到1024；这是需要实测的
取舍。18项native/graph已通过（66.64s），8shape性能验证已完成（见下表）。包SHA为
`fa112ad89c57795a70a6931c45bd84b10c8067dc628ffd1ca967f5ecba164f04`。

另一项H32搬运优化是保持同query的4KiB INT8 Q
和1KiB FP16展开weights常驻L0A，动态K/Score仅在L0B轮转。当前每个QK tile
重复装载Q，每个paired WS重复装载weights；固定L0A地址可以删除这些重复
LoadData。需要query换代M→MTE1写保护、Q/W的MTE2→MTE1就绪和初次MTE1→M
依赖，不能只删LoadData。它复用同一query的Q/weights，不假设不同query K相同。

### r27 profile计数异常与复采

首次r1采集wall为868.16µs，但20个AIC的总时间为5348–5734µs，明显不一致。
原始CSV保留；不能用这次总周期作为利用率分母。相同包、输入及命令重新采集r2，
wall为865.06µs，AIC总周期为1.36–1.55M，按1.8GHz换算与wall一致；异常未复现，
原因尚未确定。汇总脚本新增总时间超过wall 20%的异常提示，既不删除原始行，也
不自动把没有该提示的采集认定为有效。用r1/r2执行检查，分别产生提示/无提示，
并验证CSV指纹及原始行保持不变。证据：
`artifacts/qli-fused/r27/profile-counter-repeat-comparison.json`。

r2全部20AIC/40AIV均参与统计，mean时间如下；流水计数重叠，不能相加当作wall：

| 指标 | r23原布局N128 | r26紧凑N128 | r27紧凑N256 r2 |
| --- | --- | --- | --- |
| Kernel wall | 859.04 | 913.64 | 865.06 |
| AIC Cube active | 60.27 | 63.09 | 51.09 |
| AIC MTE1 | 128.41 | 145.06 | 141.89 |
| AIC MTE2 | 115.69 | 116.52 | 105.65 |
| AIC FixPipe | 120.01 | 117.58 | 108.57 |
| AIC READY等待 | 525.66 | 568.80 | 527.24 |
| gather AIV MTE2 | 537.27 | 583.03 | 539.07 |
| gather AIV scalar | 541.35 | 567.45 | 540.57 |
| top-k AIV vector | 272.66 | 272.72 | 272.66 |

N256减少了Cube/FixPipe活动时间，但完整延迟并未优于原布局；AIC仍长时间等待
连续K准备。这里只确认相关计数变化，不能把计数相加或据此假定单项优化收益。

### r29实验：同query Q/W常驻L0A

r29基于已验证r27的N256独立实现，新增默认关闭的模板参数`RESIDENT_QW`，
只由mode4/H32消费者启用。Q使用L0A字节0–4095，weights使用4096–5119，
每query各加载一次，K/Score继续按原L0B槽轮转。query换代增加独立EVENT_ID0
的M→MTE1和MTE2→MTE1配对屏障，初次QK既有MTE1→M屏障覆盖两次装载。
默认mode2模板实例不改变。r28的N512独立包已冻结保留，先完成其验证，再与
r27比较r29，避免把N宽度变化混入常驻Q/W的效果。完整构建和源码/安装指纹
一致性检查已通过，manual hooks通过；包SHA为
`beed99167a17b8e9a3b5287e942ae678ecac9639f745a4c7c634a810e5176373`。
18项native/graph已通过（55.08s），8shape性能验证已完成，未见收益（见下表）。

## H32 N512 r28完整性能结果

18项native/graph通过；8shape的fused/dense/paged三路oracle全部通过。
N512在所有shape均慢于N256，拒绝采用。单位µs：

| Shape | N512 median / P95 | 相对N256 | 对dense加速 |
| --- | --- | --- | --- |
| t128-n32771 | 1032.47 / 1039.87 | +5.64% | 1.042x |
| b4-t128-n32771 | 1058.34 / 1065.38 | +7.36% | 1.461x |
| b8-t512-n32771 | 3367.53 / 3375.35 | +9.90% | 1.023x |
| t128-n4097 | 538.40 / 539.96 | +2.11% | 0.672x |
| t128-n131075 | 1058.59 / 1063.36 | +6.94% | 2.222x |
| t512-n4097 | 1610.01 / 1619.46 | +5.94% | 0.572x |
| t512-n32771 | 3319.67 / 3325.77 | +9.71% | 1.038x |
| t512-n131075 | 3349.40 / 3356.58 | +9.62% | 2.055x |

扩大N减小了tile数，同时失去L0AB双缓冲，完整selector结果更差。
这次实验没有证明单槽是唯一根因；全核profile结果如下。
r29仍采用N256，独立验证Q/W常驻，完整样本保留在各版本artifact中。

r28全20AIC/40AIV同fixture profile无总周期异常提示。wall936.60µs，较N256
的865.06µs慢；mean Cube43.75µs、MTE1 116.89µs下降，但FixPipe131.28µs、
READY等待576.90µs上升。gather MTE2 587.33µs、scalar567.72µs，wait_ib21.97µs，
也高于N256的539.07/540.57/1.13µs；top-k vector272.72µs基本不变。

对r24/r26/r27/r28/r29实际ELF核查，选中tiling1124270594的AIV函数体均43532B，
未重定位指令字节SHA完全一致；其text偏移mod256分别为4/208/20/192/216。
证据在`artifacts/qli-fused/r29/kernel-function-bytes-comparison.json`。这证明AIV
编译体未变，不能证明运行时装载地址相同，也不能证明代码对齐是性能根因。
Cube修改伴随未修改gather计数变化，后续需单独隔离测量，不能全部归因于N大小。

## r29 Q/W常驻L0A完整性能结果

18项native/graph通过，8shape三路oracle均通过；同query的Q/W复用在本轮没有带来
完整selector收益。全部shape较r27慢，不能采用这项实验作为优化结果。单位µs：

| Shape | r29 median / P95 | 相对r27 | 对dense加速 |
| --- | --- | --- | --- |
| t128-n32771 | 1002.54 / 1008.49 | +2.57% | 1.088x |
| b4-t128-n32771 | 1019.43 / 1020.53 | +3.41% | 1.518x |
| b8-t512-n32771 | 3172.91 / 3179.29 | +3.55% | 1.083x |
| t128-n4097 | 540.18 / 542.22 | +2.45% | 0.682x |
| t128-n131075 | 1020.70 / 1021.67 | +3.11% | 2.325x |
| t512-n4097 | 1574.71 / 1584.98 | +3.62% | 0.583x |
| t512-n32771 | 3130.53 / 3133.12 | +3.46% | 1.100x |
| t512-n131075 | 3159.47 / 3163.26 | +3.40% | 2.209x |

随后在同fixture上采集全核profile，并补齐原布局r24的两个多batch性能点，
再决定保留的实验几何。当前没有任何版本达到全部冻结验收门槛，不默认启用。
