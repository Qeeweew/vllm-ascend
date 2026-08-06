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

## 2. 为什么分 vec1 / vec2（两阶段的根本原因）

epilogue 的计算天然分两段，**并行切分维度冲突**：

| 阶段 | 计算 | 并行粒度 | 为什么这样切 |
|---|---|---|---|
| **vec1** | 窗口装配 + ape + 逐通道 softmax + 加权和 | **按 D（headDim 通道）切** | 每个输出通道独立归约，40 核各算 512/groupSize 维，互不依赖 |
| **vec2** | rms_norm + rope | **按行（压缩 token）切** | rms_norm 需要整行 512 维的均方，只拥有部分维度的核算不了 |

vec1 的产出是"部分行"（每核只写了每行的 1/groupSize），因此必须经 GM（`vec1Res` workspace）中转，
`SyncAll` 全核屏障后再进入 vec2。**UB 是每核私有的，跨核数据交换只能走 GM + 屏障**——这是本算子
`SyncAll` 存在的唯一原因（每 nSize=2 个基本块一次）。

## 3. 数据流总览

```
                     ┌────────────────────── 外部 MatMulV3 ×2 ──────────────────────┐
                     │   mm_kv = x@wkv        mm_score = x@wgate   （bf16, [T,1024]）│
                     └─────────────────────────────┬─────────────────────────────────┘
                                                  ▼
        （40 个 AIV 并行，每个负责基本块的一部分 token × 一部分 D 维）
   ┌─────────────────────────── vec1（D 维并行）───────────────────────────┐
   │ for 每个基本块 (mBase=256 tokens):                                     │
   │   CopyInApe:   ape ─MTE2→ UB                                          │
   │   scoreUb = load mm_score 行段（bf16→fp32 Cast，进队列 buffer）        │
   │   kvUb    = load mm_kv    行段（同上）                                 │
   │   for 每个 slice（一组 4 token 及对齐边角）:                           │
   │     AddApeToScore:  scoreUb += ape（按组内位置）                       │
   │     SaveState:      scoreUb/kvUb → state_cache（分页，MTE3）           │
   │     ReadState:      state_cache 历史行 → 窗口左/右半（MTE2）           │
   │     PadAlign:       当前行段按窗口 8 行排布（UB→UB）                   │
   │     LoadFromWorkSpace: 窗口前驱行（跨基本块）← mm GM（bf16→fp32 Cast） │
   │     SoftmaxDN + KvMulReduceScore:  逐通道 softmax + 加权和             │
   │     CopyOutVec1Res: 部分结果 → vec1Res GM（MTE3）                      │
   │   SyncAll（每 2 个基本块）                                            │
   └────────────────────────────────┬───────────────────────────────────────┘
                                    ▼
   ┌────────────────────────── vec2（行并行）───────────────────────────────┐
   │   行段 ─MTE2→ UB → rms_norm（整行 512）→ rope（后 64）→ cmp_kv（MTE3） │
   └────────────────────────────────────────────────────────────────────────┘
```

## 4. 逐函数计算流程

### 4.0 驱动循环（kernel_perf.h `Process`）

```cpp
for (i = 0; i < loopTimes; ++i) {          // loopTimes = 本 call 的基本块数
    CalcVec1Params(...);                   // 用迭代器算出本基本块该处理哪些 token（跨 batch、跳过无效）
    ComputeVec1(vec1Info);                 // 本核在本基本块的任务
    UpdateVec2Info(...);                   // 累加压缩行数，供 vec2 使用
    if (IsNeedSyncAll(i)) {                // 每 nSize=2 块 + 最后一块
        SyncAll();                          // 全核屏障：vec1Res 全局可见
        if (vec2Info.dealScSize > 0) ComputeVec2(vec2Info);  // 处理累积的压缩行
    }
}
```

- `SkipOneLoop`（tools.h 的迭代器）：把 M 个 token 切成基本块（mBase=256），跨 batch 边界时按
  `start_pos` 对齐到 cmpRatio 组边界，跳过 seqUsed 外的 gap 行。
- 每块 64 个压缩单元（256/4），40 核均分 → 每核每块 1~2 个 tc，**工作粒度很小**。

### 4.1 vec1：核间切分（`ComputeVec1` → `SplitCoreV1`）

四步决策：

1. **`CalcGroupInfo`**：按 `dealTcNum` 与 40 核数选 D 切分粒度
   `dBaseSize = headDim / min(FloorPow2(40), CeilPow2(CeilDiv(40, dealTcNum)))` ∈ {16..512}；
   `groupSize = headDim/dBaseSize`（D 维每组的核数），`groupNum = min(40/groupSize, dealTcNum)`（组数 = 行方向并行的基本单元数）。
2. **`CalcTaskDistribution`**：按 `blockIdx` 负载均衡分配 tc：`dealTcSize`（本核处理的 tc 数）与
   `preDealTcSize`（本核起点之前的 tc 数，用于定位 token 起点）。
3. **`UpdateIteratorState`**：用 slice 迭代器把"前序 tc"推进一遍，得到本核的 `(curBStart, curSStart, dealSeqStartIdx)`
   起点与 `curCompressedCnt`（前序已产出的压缩行数）。
4. **`CalcTilingStrategy`**：`maxDealColNum = 32K/(cmpRatio*coff*4)`，决定 `tcSplitSize`（一个基本块装几个 tc）
   与 `dSplitSize`（D 维一次算多少）。

主循环：`dLoop`（D 分块）→ `tcLoop`（tc 分块，每块调 `DealVec1BaseBlock`）。

### 4.2 vec1：单个基本块（`DealVec1BaseBlock`）

```
originSliceInfo = 本块起始 slice（含 sIdx/dealedSeqCnt 等）
statisticInfo   = 迭代 needDealTcSize 个 tc 的统计（dealSeqCnt=要加载的 token 数,
                  compressScCnt=要输出的压缩行数）
scoreLocal = tmpBuff1, kvLocal = tmpBuff2   （窗口缓冲，32K/64K）
OverLapScoreKv(...)                          // 装配 8 行窗口（见 4.3）
SoftmaxDN(scoreLocal)                        // 逐通道 softmax（8 行一组）
KvMulReduceScore → comperssoredUb            // p·kv 逐通道求和 → 部分压缩行
CopyOutVec1Res → vec1Res GM                  // 写到 (v1v2DbIdx*dbSize + compressedCnt_*headDim + dStartIdx)
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
    （尾部 MTE2_V flag：窗口装配完成 → 可被 V 计算消费）
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
  半边（D_L），行数 = `min(sIdx, cmpRatio)`。原算子这里读 cacheTc（已含 ape 的缓存），split 直接读裸 mm GM，
  因此需要额外的 `AddSingleApeToScore` 给左半补 ape。
- **SaveState 的分页写**：`stateOffset = blockTable[blockId]*stride0 + remainRow*2*coff*headDim
  + stateIdx*coff*headDim + dStartIdx`，blockId/remainRow 由绝对 seq（含 start_pos）算出；stateIdx 0=kv、1=score。

### 4.4 vec2（`ComputeVec2` → `DealVec2BaseBlock`）

```
SplitCoreV2          // 把累积的压缩行按核均分（行并行）
for 每个行块:
    vec1ResUb = vec2InputGm[行段] (MTE2)          // 读 vec1 的部分和
    MultRowRmsNorm：rms = sqrt(mean(x²)+eps); x/rms*norm_weight
    CalRope：后 ropeHeadDim 维（用 rope_sin/cos 按行索引）
    CopyFinalResultOut → cmp_kv GM（TH 布局按 batch 紧凑）
```

## 5. 同步模型（与原始 Compressor 保持一致 + 两处新增）

原 `Compressor` 的同步纪律（逐字节沿用）：

- **队列自动同步**：`inputQue1`（VECIN：EnQue=MTE2→V，Free/Alloc=V→MTE2）、`outputQue1`
  （VECOUT：V→MTE3）覆盖常规的"搬运→计算→搬出"与 buffer 复用；
- **3 对定向 flag**（`OverLap` 内）：`V_MTE2`（SaveState 后，V 计算 → ReadState 的 MTE2 写）、
  `MTE3_MTE2`（SaveToWorkSpace 的 MTE3 写 → LoadFromWorkSpace 的 MTE2 读，split 中保留）、
  `MTE2_V`（OverLap 尾部，窗口装配完成 → 后续 V 计算）。

split 新增（**仅此 4 处**，`csrc/attention/compressor_epilogue/op_kernel/arch32/compressor_epilogue_block_vec_perf.h`）：

| 位置 | 同步 | 原因 |
|---|---|---|
| `FromWokrSpaceToUb`：stage copy(MTE2) → Cast(V) 之间 | 一对 `MTE2_V` flag | 裸 DataCopy 与向量混用的标准同步（与原算子纪律一致） |
| `FromWokrSpaceToUb`：Cast 之后 | **`PipeBarrier<PIPE_ALL>`** | Cast 整块读写队列 buffer（读后半段 stage、写全 buffer）；队列 free 事件只保证 V 写完成，不保证 Cast 的读排空。实测仅靠 flag 会随机行损坏 |
| `LoadFromWorkSpace`：GM copy → 原地 Cast 之间 | 一对 `MTE2_V` flag | 同上，裸 copy/向量混用 |
| `OverLap`：SaveState 之前 | **`PipeBarrier<PIPE_ALL>`** | 排空 V（Cast/AddApe）后再让 SaveState 的中转 copy 读 srcLocal 及后续 ReadState/PadAlign 写窗口。实测去掉后最后一个 slice 的窗口确定性损坏 |

> 曾被怀疑、经二分证伪后**不需要**的屏障：scoreUb→kvUb 复用点、LoadFromWorkSpace 尾部 V_MTE2。
> 均由队列 free 事件覆盖。

## 6. 为什么"过于复杂"（复杂度来源）

1. **vec1/vec2 两段式 + 16 次 `SyncAll`**：rms_norm 的全行归约迫使行并行阶段必须等 D 并行阶段把部分和
   写进 GM。这是功能性的，但每 2 个基本块一次全核屏障让固定开销很大。
2. **每基本块的工作粒度太小**：mBase=256 → 每块 64 tc / 40 核 = 每核 1~2 tc。而每个 tc 要摊付
   score+kv 两次 load（各含 stage copy + flag + Cast + PIPE_ALL）、每 slice 的 SaveState/ReadState/
   PadAlign 队列往返 + flag、以及 vec1Res 写出。**固定同步开销 / 工作量 的比例非常高**（msprof：
   vec 实际只占 21.5%，scalar 等待占 ~60%）。
3. **窗口装配的边角处理**：slice 的 headHolder/tailHolder 对齐、跨基本块前驱行的 LoadFromWorkSpace、
   score 左半缺 ape 的补加（AddSingleApeToScore）、首组 DuplicateFirstBlock……这些是 DSA 语义
   （overlap 窗口、分页 state、start_pos 对齐）的必然产物，原 fused 算子同样有，不是 split 引入的。
4. **寄存器/UB 预算约束**：inputQue1 只有 32K（单 buffer），限制了一次能装的 token 行数；apeBuf 32K、
   tmpBuff1/2 32K/64K 都是各自阶段的专用缓冲——buffer 复用几乎不可行，导致每阶段都要完整搬入搬出。

## 7. 可优化方向（按性价比排序）

| 方向 | 预期收益 | 说明 |
|---|---|---|
| 加大每核每块工作量（mBase 256→1024+，配套加大 inputQue1/合并 dLoop） | 减少基本块数 ×4，同步开销线性下降 | 受 UB 预算约束，需重新配平各 buffer |
| 合并 score/kv 的加载（同一次行段一次读完两个 mm GM，共享 stage/Cast/barrier） | 每块省一半 load 开销（2 次→1 次） | mm_kv/mm_score 是同一批 token，行寻址一致 |
| 加大 nSize（2→4/8），减少 SyncAll 次数 | 全局屏障 16 次 → 4~8 次 | vec1Res workspace 相应翻倍（内存充足） |
| 独立 stage buffer 替代"队列 buffer 后半段" | 去掉 post-Cast `PIPE_ALL`（用定向 flag 替代） | 需验证 flag 是否足够（当前实测不足，需重试） |
| vec2 直接消费 vec1 的 UB 结果（跨核 reduce 替代 GM 中转） | 省 vec1Res 流量与一次 SyncAll 语义 | 910B 无跨核 UB，需改用 GM 原子/片上转发，工程量大 |

## 8. 关键文件

- `op_kernel/arch32/compressor_epilogue_kernel_perf.h`：驱动循环（Process/CalcVec1Params/SkipOneLoop/UpdateVec2Info/SyncAll 节奏）
- `op_kernel/arch32/compressor_epilogue_block_vec_perf.h`：vec1/vec2 全部计算（ComputeVec1/SplitCoreV1/DealVec1BaseBlock/OverLapScoreKv/OverLap/SaveState/ReadState/PadAlign/LoadFromWorkSpace/SoftmaxDN/KvMulReduceScore/CopyOutVec1Res/ComputeVec2/CalRope/CopyFinalResultOut）
- `op_kernel/arch32/compressor_epilogue_tools.h`：slice 迭代器（基本块/tc/slice 划分、对齐与 gap 处理）
- `op_host/arch32/compressor_epilogue_tiling.cpp`：mBase/nSize/dBaseSize/workspace 等 tiling 参数
