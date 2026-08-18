# Compressor Epilogue c128 问题分析与修复状态

> 对象：DeepSeek V4 DSA Compressor 的 split 实现（MatMulV3 + `compressor_epilogue`）。
> ratio=128（下称 c128）时 epilogue 存在问题；ratio=4（c4）完全正常。
> 测试入口：`benchmarks/ops_profiling/bench_compressor_accuracy.py --ratio 128`。

## 背景：两种 compressor 实现

| | fused（`compressor`） | split（MatMulV3 + `compressor_epilogue`） |
|---|---|---|
| 架构 | cube mm → **workspace** → vec1(d 分块产片) → **vec1Res workspace** → vec2(rms_norm+rope) | MatMulV3 → epilogue（**单核内串行**：装配→softmax→压缩→rms_norm→rope→输出，无 workspace） |
| rms_norm 位置 | **vec2**，从 workspace 读完整行 | **vec1 内**，直接对 compressedUb 做 |
| c128 | 能跑 | **有问题（本文）** |

配置语义（`RATIO_CONFIG`）：c4 = {coff:2(overlap), state_block:8}；c128 = {coff:1, state_block:32}。
c128 触发：`VLLM_ASCEND_DSA_COMPRESSOR_SPLIT=1` + ratio=128。

## 三个 reduce 维（理解问题的前提）

| 算子 | reduce 维 | d（headDim=512）是否独立 |
|------|-----------|--------------------------|
| softmax（`ColumnSoftMax`） | over 窗口行 `coff*cmpRatio`（c128=128） | **独立 → d 可分块** |
| 压缩（`ColumnSum`） | over 窗口行（128） | **独立 → d 可分块** |
| rms_norm（`RmsNorm`） | over headDim（512） | **必须完整行 → d 不可分块** |

c128 窗口 = `coff*cmpRatio` 行 × headDim 列 = 128×512×4B = **256KB**，超过 UB（192KB）与所有单 buffer（`tmpBuff1 32K`/`tmpBuff2 64K`/`apeBuf 16K`）。c4 窗口只有 8 行（16KB），无此问题。

## 问题 1：UB 越界崩溃（已修）

- **现象**：`VEC instruction error: the ub address out of bounds, fixp_error 0x6000022, blk:10`。
- **根因**：`CalcTilingStrategy` 在 `maxDealColNum = 32K/(cmpRatio*coff*4B) = 64 < headDim=512` 时走 d 分块（`dSplitSize=64`）。`CopyInApe` 把 ape 整片 `[coff*cmpRatio × dSplitSize]` = 128×64×4B = **32KB 拷进 `apeBuf`(16K)** → 越界。
- **修复**：`CalcTilingStrategy` 的 d 分块分支加 ape 预算约束
  `apeMaxDealColNum = 16K/(cmpRatio*coff*4B)`，c128 → `dSplitSize=32`（ape=128×32×4B=16KB 恰好）。
  同时 `CopyInApe` 从连续拷贝改为按行 stride 拷贝（`DataCopyAlignGmToUb(..., srcStride=headDim, dstStride=dDealSize)`）——c128 d 分块后 `dDealSize < headDim`，必须按行 stride 取列否则跨行错位（c4 `dDealSize==headDim` 时两式等价，不受影响）。
- **状态**：c128 epilogue 不再崩溃，输出确定（两次运行 diff=0）。

## 问题 2：d 分块破坏 rms_norm（待修，数值错误）

- **现象**：c128 下 `fused vs epilogue` 真实压缩行 rel ≈ **2.6**（264%），且 ape=0 时依旧 → 与 ape 无关。state_cache 一致（rel 0.0024 PASS），错在 cmp_kv 主输出。
- **根因**：epilogue 是**单核内串行、无 workspace**，rms_norm 在 `DealVec1BaseBlock` 内对 `compressedUb` 做，`col=headDim=512`：
  ```cpp
  KvMulReduceScore(..., compressedUb, ..., dDealSize);   // d 分块时 compressedUb 只有 [scCnt × dSplitSize=32]
  rmsNormParams.col = constInfo_.headDim;                 // =512
  RmsNorm(compressedUb, ..., rmsNormParams);              // 对 32 列的片按 512 列做 rms_norm → 错
  ```
  d 分块后压缩片只有 32 列，rms_norm 却按完整 512 列求 mean(x²) → 数值错误。
- **为什么 fused 没问题**：fused 把 rms_norm 挪到 **vec2**，vec1 d 分块产的片写进 `vec1ResGm_` workspace 的完整行布局（`CopyOutVec1Res` 按 `stride=headDim` 写对应列），vec2 再从 workspace **读完整行**做 `RmsNorm(col=512)`。workspace 是"d 片 → 完整行"的桥梁，所以 fused 的 d 分块正确。epilogue 没有这个桥梁。

## 修复方向（待决策）

- **A. online softmax（窗口行分片流式）**：保持 d=512 不分块，把 128 窗口行分片（如 8 行/片）流式处理 + online softmax 合并 + 压缩部分和累加。rms_norm 天然拿到完整行。**难点**：窗口装配（`OverLapScoreKv`）的 state_cache 分页 / overlap 左右半交错 / `slice iterator` 都是围绕"完整窗口"设计的，没有"窗口行"粒度接口，需重写整套 helper；且 `CopyInMm` 本身 `dealSeqCnt` 也装不下。复杂度高。
- **B. d 分块 + 攒完整行再 rms_norm（借鉴 fused）**：保留 d 分块（窗口/ape/CopyInMm 都装得下），d 分块循环内只做 装配→softmax→压缩 并把各 d 片**攒进一块 UB 拼成完整行**（c128 scCnt 小，c128 常见 dealTcSize=1 时仅 2KB，单核内无需 GM workspace），d 循环结束后统一 `rms_norm(col=512)+rope+输出`。**难点**：重组 `ComputeVec1` 的 d/tc 循环、UB 布局重排、rope cos/sin 与输出行游标（TH/BSH）时机的迁移；但不动装配逻辑，比 A 简单，且与 fused 已验证语义一致。

## 当前代码状态

- `compressor_epilogue_block_vec_perf.h`：
  - `CopyInApe`：stride 拷贝（正确，任何修复方向的前置）。
  - `CalcTilingStrategy`：d 分块 + ape 预算约束（dSplitSize=32），**修好崩溃但引入 rms_norm 数值错误**（问题 2）。
- 即：**c128 当前"能跑不崩、确定性 OK，但 cmp_kv 数值错误"**，不可用于生产精度路径。c4 完全正常（走 else 分支，回归 ALL PASS）。

## 验证方法

```bash
cd benchmarks/ops_profiling
# c4 全量回归（应 ALL PASS）
ASCEND_RT_VISIBLE_DEVICES=0 python bench_compressor_accuracy.py
# c128 诊断（当前：epilogue 不崩但 fused vs epi rel≈2.6 FAIL；修复问题 2 后应转 PASS）
ASCEND_RT_VISIBLE_DEVICES=0 python bench_compressor_accuracy.py --c128-diagnose
```
