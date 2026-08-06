# CompressorEpilogue 算子设计与计算流程

> 本文档描述 `compressor_epilogue`（`torch.ops._C_ascend.compressor_epilogue`）在 910B4-1（arch32, AIV-only）
> 上的计算流程与同步模型。它是 DeepSeek-V4 DSA `Compressor` 融合算子拆分的产物：cube GEMM（`x@wkv`、
> `x@wgate`）由外部两个 `MatMulV3`（`F.linear`）完成，本算子只做纯 AIV 的向量部分（epilogue）。

## 1. 算子职责

输入（均为本 call 内的数据，TH 布局展平）：

- `mm_kv`/`mm_score`：外部 GEMM 输出 `[totalTokens, coff*headDim]`（= `[T, 2*512]`）bf16/fp16，**不含 ape**
- `state_cache`：`[maxBlocks, 8, 2*coff*headDim]` fp32，分页 KV/score 历史（**in-place 读写**）
- `ape`：`[cmpRatio, coff*headDim]` fp32，窗口内按组内位置加的偏置
- `norm_weight`、`rope_sin/cos`、`state_block_table`（分页表）、`cu_seqlens`、`start_pos`、`seqused`

输出：

- `cmp_kv`：`[min(T, T/cmpRatio + B), headDim]` bf16
- `state_cache`（in-place）：本 call 每个 token 的 `[kv|score]`（coff0/coff1 两个半边），供 sparse_attn 后续使用

语义（coff=2, cmpRatio=4 时，已验证与 torch 参考一致）：

- 第 k 个压缩 token 的窗口 = **8 行**：`[前一组 4 token 的 coff0 | 当前组 4 token 的 coff1]`
- `score' = mm_score + ape`（ape 按窗口内组内位置行加）
- 对 8 行**逐通道**做 softmax（ColumnSoftMax），`cmp[c] = Σ₈ p·kv`（ColumnSum）
- 压缩结果行做 rms_norm（整行 512 维）+ rope（后 64 维）→ `cmp_kv`

## 2. 架构：单阶段完全串行（无跨核同步）

历史上有两个架构，均已被替换：

- **v1（两阶段 + SyncAll）**：vec1（按 D 维并行，部分行写 GM workspace `vec1Res`）+ 每 nSize=2 基本块一次
  `SyncAll` 全核屏障 + vec2（按行并行读 `vec1Res` 做 rms_norm/rope）。rms_norm 需要整行 512 维，而 vec1
  按 D 切分时每核只有部分行，因此必须经 GM + 全局屏障交换数据。
- **v2（当前，完全串行）**：**每核独占完整 `headDim` 维（行并行）**。窗口装配、softmax、加权和、rms_norm、
  rope、写 `cmp_kv` 全部在核内完成，**无任何跨核数据依赖** → 删掉 `SyncAll`、`vec1Res` workspace、vec2 阶段、
  双缓冲簿记。kernel 退化为纯流水，只有队列自动同步 + 定向 flag。

行并行成立的依据（`CalcGroupInfo`）：

- 40 个 AIV 按 token（tc）均分，每核处理若干**完整行**；
- 每核读写 state_cache 都只落在自己负责的行/列上（窗口 ReadState 与 SaveState 的列段一致）；
- 输出行号 = 本核 `compressedCnt_`（全局压缩行号，tc 按序分核故连续）。

性能实测（M=8192 prefill，msprof）：

| 版本 | epilogue | prefill 总 | 说明 |
|---|---|---|---|
| fused 单算子 | 1983.3us（含 GEMM） | 1983.3us | GEMM L1 复用瓶颈 50.8% SOL |
| v1（16×SyncAll） | 454.98us | 1568.7us | wait_id14（事件等待）= 103.4us |
| **v2（完全串行）** | **248.74us** | **1380.1us** | **wait_id14 → 5.7us（-94%）** |

v2 的 `aiv_vec` 占用 38.3%（91us），`mte2/mte3` 各 ~19%，剩余主要是 2 处 PIPE_ALL + flag 的指令等待。

## 3. 数据流总览

```
                     ┌────────────────────── 外部 MatMulV3 ×2 ──────────────────────┐
                     │   mm_kv = x@wkv        mm_score = x@wgate   （bf16, [T,1024]）│
                     └─────────────────────────────┬─────────────────────────────────┘
                                                  ▼
        （40 个 AIV 行并行：每核处理若干 tc 的完整 512 维行，全程无跨核通信）
   ┌─────────────────────────────── 单阶段流水 ───────────────────────────────┐
   │ for 每个基本块 (mBase=256 tokens，每核 1~2 tc):                           │
   │   CopyInApe:   ape ─MTE2→ UB                                            │
   │   scoreUb = load mm_score 行段（bf16→fp32 Cast）                         │
   │   kvUb    = load mm_kv    行段                                          │
   │   for 每个 slice（一组 4 token 及对齐边角）:                             │
   │     AddApeToScore:  scoreUb += ape（按组内位置）                         │
   │     SaveState:      scoreUb/kvUb → state_cache（分页，MTE3）             │
   │     ReadState:      state_cache 历史行 → 窗口左右半（MTE2）               │
   │     PadAlign:       当前行段按窗口 8 行排布（UB→UB）                      │
   │     LoadFromWorkSpace: 窗口前驱行（跨基本块）← mm GM（bf16→fp32 Cast）    │
   │     SoftmaxDN + KvMulReduceScore:  逐通道 softmax + 加权和 → 压缩行（核内）│
   │     FinishCompressedRows:  rms_norm（整行 512）→ rope（后 64）→ cast      │
   │                           → 直接写 cmp_kv（TH 紧凑 / BSH 逐 batch）       │
   └──────────────────────────────────────────────────────────────────────────┘
```

## 4. 逐函数计算流程

### 4.0 驱动循环（kernel_perf.h `Process`）

```cpp
for (i = 0; i < loopTimes; ++i) {          // loopTimes = 本 call 的基本块数
    CalcVec1Params(vec1Info, batchInfo, i); // 用迭代器算出本基本块该处理哪些 token（跨 batch、跳过无效）
    ComputeVec1(vec1Info);                  // 本核在本基本块的任务（含 rms_norm/rope/输出）
}
```

- `SkipOneLoop`（tools.h 的迭代器）：把 M 个 token 切成基本块（mBase=256），跨 batch 边界时按
  `start_pos` 对齐到 cmpRatio 组边界，跳过 seqUsed 外的 gap 行。

### 4.1 核间切分（`ComputeVec1` → `SplitCoreV1`）

四步决策：

1. **`CalcGroupInfo`**：**强制 `dBaseSize = headDim`**（每核全维），`groupSize=1`，`groupNum = min(40, dealTcNum)`。
2. **`CalcTaskDistribution`**：按 `blockIdx` 负载均衡分配 tc：`dealTcSize`（本核处理的 tc 数）与
   `preDealTcSize`（本核起点之前的 tc 数，用于定位 token 起点）。
3. **`UpdateIteratorState`**：用 slice 迭代器把"前序 tc"推进一遍，得到本核的 `(curBStart, curSStart, dealSeqStartIdx)`
   起点与 `curCompressedCnt`（前序已产出的压缩行数）。
4. **`CalcTilingStrategy`**：`maxDealColNum = 32K/(cmpRatio*coff*4)`，决定 `tcSplitSize`（一个基本块装几个 tc）
   与 `dSplitSize`（D 维一次算多少，= 512 全维）。

主循环：`dLoop`（D 分块，恒 1 次）→ `tcLoop`（tc 分块，每块调 `DealVec1BaseBlock`）。

> 输出游标（仅 BSH 布局输出映射使用）：`ComputeVec1` 首次进入时按核起点 slice 初始化
> `(OutputBStartIdx, OutputSStartIdx)`，之后每次产出由 `UpdateOutputIdx` 推进。TH 布局直接用
> `compressedCnt_`（全局压缩行号）作为输出偏移，不需要游标。

### 4.2 单个基本块（`DealVec1BaseBlock`）

```
originSliceInfo = 本块起始 slice（含 sIdx/dealedSeqCnt 等）
statisticInfo   = 迭代 needDealTcSize 个 tc 的统计（dealSeqCnt=要加载的 token 数,
                  compressScCnt=要输出的压缩行数）
scoreLocal = tmpBuff1, kvLocal = tmpBuff2   （窗口缓冲，32K/64K）
OverLapScoreKv(...)                          // 装配 8 行窗口（见 4.3）
SoftmaxDN(scoreLocal)                        // 逐通道 softmax（8 行一组）
KvMulReduceScore → compressedUb(tmpBuff1)    // p·kv 逐通道求和 → 核内完整压缩行
FinishCompressedRows(compressedUb)           // rms_norm + rope + cast + 直接写 cmp_kv
compressedCnt_ += compressScCnt
```

### 4.3 窗口装配（`OverLapScoreKv` + `OverLap`）——最复杂的部分

对 score 和 kv 各跑一遍 `OverLap`（score 多一个 AddApeToScore）：

```
FromWokrSpaceToUb(scoreUb/kvUb)   // mm GM 行段 → UB（bf16 暂存 → Cast fp32）
for 每个 overlap slice:
    AddApeToScore(srcLocal)                       // score 专用：按组内位置加 ape
    SaveState(srcLocal → state_cache GM)          // 本 slice 的 valid 行写分页状态
    ReadState(state_cache → dstLocal)             // 窗口左右半的历史行：
                                                  //   右半: headHolder 行（bStartPos 之前的尾行）
                                                  //   左半: 前一组 4 行（coff0），首组用 DuplicateFirstBlock 复制
    PadAlign(dstLocal, srcLocal)                  // 把 srcLocal 的连续行重排成窗口 8 行 × [coff0|coff1]
    LoadFromWorkSpace(dstLocal)                   // 窗口左半的前驱 4 行：跨基本块时从 mm GM 读（bf16→Cast）
```

关键点：

- **slice 是什么**：按 `start_pos` 对齐的"组 + 边角"划分。一个 slice 可能只有 headHolder（不足 4 的
  历史尾行）、valid（本组 token）、tailHolder（不足 4 的尾行）中的若干部分。`dealTcSize`（本 slice 覆盖
  几个组）、`compressTcSize`（其中几个组能产出压缩行，尾组不满不产出）。
- **窗口行来源**：右半（当前组 coff1）= srcLocal 当前行；左半（前一组 coff0）= 本 slice 内更早行
  （PadAlign 从 srcLocal 搬）或跨 slice/基本块的前驱行（LoadFromWorkSpace 从 mm GM 或 UB 搬）。
- **`FromWokrSpaceToUb` 的 mm GM 寻址**：行段起点 = `(cuSeqlens[bIdx] + sIdx) * coff*headDim + dStartIdx`，
  行 stride = `headDim`（coff0/coff1 交错），一次读 `dealSeqCnt*coff_` 行 × `dDealSize` 列。
- **`LoadFromWorkSpace` 的 mm GM 寻址**：前驱 token 行 stride = `coff*headDim`（每 token 一行），只取 coff0
  半边（D_L），行数 = `min(sIdx, cmpRatio)`。由于 split 直接读裸 mm GM（不含 ape），左半行还需要
  `AddSingleApeToScore` 补 ape。
- **SaveState 的分页写**：`stateOffset = blockTable[blockId]*stride0 + remainRow*2*coff*headDim
  + stateIdx*coff*headDim + dStartIdx`，blockId/remainRow 由绝对 seq（含 start_pos）算出；stateIdx 0=kv、1=score。

### 4.4 压缩行收尾（`FinishCompressedRows`，v2 新增）

```
RmsNorm(compressedUb, ..., normWeightUb, tmpUb, {reciprocalD, normEps, scCnt, headDim})   // 整行归范
SingleCalRope(..., compressedCnt_)    // 只对后 ropeHeadDim 维做 rope，sin/cos 按全局压缩行号取
Cast(outputUb, compressedUb, CAST_RINT, scCnt*headDim)   // fp32 → X_T（bf16/fp16）
CopyFinalResultOut(outputUb, scCnt)   // TH: cmpKvOutGm_[compressedCnt_*headDim]；BSH: 游标逐 batch
```

buffer 生命周期（均复用，无新增 UB）：

- 压缩行写入 `tmpBuff1`（scoreLocal 窗口在 KvMulReduceScore 的 Mul 之后即废弃）；
- rms_norm temp 用 `tmpUb`（tmpBuff2 后半 32K）；
- rope 的 sin/cos 经 `inputQue1`（mm load 已结束）读入，fp32 转换用 `tmpBuff2`（kvLocal 已废弃）；
- 输出经 `outputQue1`（X_T，16K）。

## 5. 同步模型

完全串行化后，**无 SyncAll、无 workspace、无 vec2 阶段**。剩余同步：

- **队列自动同步**：`inputQue1`（VECIN：EnQue=MTE2→V，Free/Alloc=V→MTE2）、`outputQue1`
  （VECOUT：V→MTE3）覆盖常规的"搬运→计算→搬出"与 buffer 复用；
- **3 对定向 flag**（`OverLap` 内，沿用原 Compressor）：`V_MTE2`（SaveState 后，V 计算 → ReadState 的 MTE2 写）、
  `MTE3_MTE2`（SaveToWorkSpace 的 MTE3 写 → LoadFromWorkSpace 的 MTE2 读）、
  `MTE2_V`（OverLap 尾部，窗口装配完成 → 后续 V 计算）；
- **2 对新增 `MTE2_V` flag**（裸 DataCopy 与向量混用）：`FromWokrSpaceToUb` 的 stage copy→Cast、
  `LoadFromWorkSpace` 的 GM copy→原地 Cast；
- **2 处 `PipeBarrier<PIPE_ALL>`**（逐一二分实证必需，去掉必坏）：
  - `FromWokrSpaceToUb` Cast 之后：Cast 整块读写队列 buffer，队列 free 事件只保证 V 写完成、不保证 Cast
    读排空；
  - `OverLap` SaveState 之前：排空 V（Cast/AddApe）后再让 SaveState 的中转 copy 读 srcLocal。

> 曾存在、经二分证伪删除的屏障：scoreUb→kvUb 复用点、LoadFromWorkSpace 尾部 V_MTE2（队列 free 事件已覆盖）、
> 以及 v1 的全部 16 次 SyncAll（行并行后无跨核数据依赖，整段删除）。

## 6. 复杂度来源（现状评估）

1. **行并行后无全局同步**：复杂度大头（两阶段 + SyncAll + 双缓冲）已消除，driver 是平凡循环。
2. **窗口装配的边角处理**：slice 的 headHolder/tailHolder 对齐、跨基本块前驱行的 LoadFromWorkSpace、
   score 左半缺 ape 的补加（AddSingleApeToScore）、首组 DuplicateFirstBlock、分页 state 的 blockId/remainRow
   计算——这些是 DSA 语义（overlap 窗口、分页 state、start_pos 对齐）的必然产物，fused 原算子同样有。
3. **UB 预算约束**：inputQue1 只有 32K（单 buffer），限制了一次能装的 token 行数；apeBuf 32K、tmpBuff1/2
   32K/64K 都是各阶段的专用缓冲。当前 181K/192K 已配平，压缩行收尾全部复用既有 buffer。

## 7. 性能现状与剩余优化方向

M=8192 prefill 实测：epilogue **248.74us**（aiv_vec 38.3%），prefill 总 **1380.1us**（GEMM 2×564.6us 已是
主导，86.7% SOL）。epilogue 相对内存下限（~66us）还有 ~3.8× 差距，剩余可优化：

| 方向 | 预期收益 | 说明 |
|---|---|---|
| 降低 2 处 PIPE_ALL + flag 的指令等待（~58us） | 中等 | 尝试独立 stage buffer 替代"队列 buffer 后半段"，用定向 flag 替换 PIPE_ALL（需重验） |
| 加大 inputQue1（32K→64K），减少行段 load 次数 | 中等 | 需从 apeBuf/tmpBuff1 挪 UB 预算，重新配平 |
| 合并 score/kv 的加载（同一行段一次读完两个 mm GM） | 中低 | 共享 stage/Cast/flag，每块省一半 load 开销 |
| icache（miss 4.1%） | 低 | kernel 已变小，可再验证 code layout |

## 8. 关键文件

- `op_kernel/arch32/compressor_epilogue_kernel_perf.h`：驱动循环（Process/CalcVec1Params/SkipOneLoop）
- `op_kernel/arch32/compressor_epilogue_block_vec_perf.h`：全部计算（ComputeVec1/SplitCoreV1/DealVec1BaseBlock/
  OverLapScoreKv/OverLap/SaveState/ReadState/PadAlign/LoadFromWorkSpace/SoftmaxDN/KvMulReduceScore/
  FinishCompressedRows/CopyFinalResultOut）
- `op_kernel/arch32/compressor_epilogue_tools.h`：slice 迭代器（基本块/tc/slice 划分、对齐与 gap 处理）
- `op_host/arch32/compressor_epilogue_tiling.cpp`：mBaseSize/tcSplitSize/workspace（已无 vec1Res）等 tiling 参数
