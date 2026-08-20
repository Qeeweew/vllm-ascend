# DeepSeek V4 DSA Compressor 算子优化报告

**测试日期**：2026-08-07  
**测试环境**：Ascend910B4-1（单 die，20 AIC + 40 AIV，HBM 64GB，L2 96MB），CANN 9.0.0，测试卡 **NPU 4**（空闲）  
**硬件基准**：BF16 cube 245.76 TFLOPS（20 AIC）、HBM 1.6 TB/s、机器平衡点 153.6 FLOP/B  
**测量方法**：msprof **设备侧 kernel `task_time` 总和**（9 次 launch 平均）；权重 concat 计时外一次性完成；decode 按 CUDA graph 场景（launch 开销已隐藏）  
**脚本**：`benchmarks/ops_profiling/bench_compressor_kernel_time.py`；数值回归 `test_compressor_split.py`

---

## 1. 背景与动机

DeepSeek V4 稀疏注意力（DSA）用**压缩 token** 代替全部历史 KV 做 indexer/topk，把注意力 KV 规模缩小 `cmp_ratio` 倍。`Compressor` 是 DSA 前处理：**每 `cmp_ratio` 个 token 的 KV 压缩成 1 个**，同时维护分页历史（state_cache）供下游 sparse_attn 使用。它在每个 prefill chunk / decode 步执行，是 DSA 的固定开销。

**优化动机（fused 实测，prefill M=8192，msprof op 流水线）**：

| AIC 侧指标 | 数值 | 诊断 |
|---|---|---|
| aic_cube_time | 1180us（**58.5%**） | cube MAC 大量时间等数据 |
| **aic_mte2_time** | **1636us（81.2%）** | **GM→片上搬入接近饱和** |

根因：GEMM `[M,7168]×[7168,1024]` 输出 N=1024 太小 → A 矩阵（117MB）L1 复用差 → 每次计算重拉 A → MTE2 饱和拖垮 cube → fused 总 **2016us，有效算力 119 TFLOPS = 48.6% SOL**。对照 N=2048 的独立 MatMulV3 达 97.4% cube——瓶颈在 N 太小而非 GEMM 本身。

## 2. 算子输入输出

### 2.1 输入（Compressor，M 个 token，B 个序列，TH 布局）

| 名称 | shape | dtype | 含义 |
|---|---|---|---|
| x | `[M, hidden]`（=7168） | bf16/fp16 | 原始 token 隐藏状态 |
| wkv | `[coff*head_dim, hidden]` | bf16/fp16 | KV 压缩权重 |
| wgate | `[coff*head_dim, hidden]` | bf16/fp16 | score 压缩权重 |
| state_cache | `[num_blocks, 8, 2*coff*head_dim]` | fp32 | 分页 KV/score 历史（**in-place 读写**） |
| ape | `[cmp_ratio, coff*head_dim]` | fp32 | 窗口位置偏置 |
| norm_weight | `[head_dim]` | bf16/fp16 | rms_norm 系数 |
| rope_sin / rope_cos | `[min(M, M/cmp_ratio+B), rope_head_dim]` | fp32 | 旋转编码系数（按压缩 token 索引） |
| state_block_table | `[B, blocks]` | int32 | 分页表（token 块 → state 块） |
| cu_seqlens | `[B+1]` | int32 | 序列边界（TH 布局） |
| seqused | `[B]` | int32 可选 | 每个序列的有效 token 数（跳过 gap） |
| start_pos | `[B]` | int32 | 每个序列的绝对位置（窗口组对齐） |
| rope_head_dim / cmp_ratio / coff / norm_eps / rotary_mode / cache_mode | — | — | 属性 |

### 2.2 输出

| 名称 | shape | dtype | 含义 |
|---|---|---|---|
| cmp_kv | `[min(M, M/cmp_ratio+B), head_dim]` | bf16/fp16 | 压缩后的 KV（每压缩 token 一行） |
| state_cache | 同输入（in-place） | fp32 | 本 call 每 token 的 `[kv\|score]` 写入历史，供后续 sparse_attn / 下个 call 读取 |

### 2.3 当前优化后（split）的算子划分

- `MatMulV3`（`F.linear`）×2：`mm_kv = x @ wkv^T`、`mm_score = x @ wgate^T`，各输出 `[M, coff*head_dim]` bf16；
- `compressor_epilogue`：输入 `mm_kv`、`mm_score` + 其余张量，输出 `cmp_kv` + in-place `state_cache`。

## 3. 计算流程（数学公式，C4A 与 C128A 分叉）

符号约定：`X ∈ ℝ^{M×H}` 输入，`r = cmp_ratio`，`c = coff`，`N = head_dim`，`s` = 序列 token 数，
第 `i` 个压缩 token（`i = 1..s/r`），`[1]_{1×r}` = 全 1 行向量，`⊙` = Hadamard 积。

### 3.1 投影（GEMM）

$$
\textbf{if } r = 4 \text{ (C4A):} \qquad
[kv^a, score^a] = X\,[W^{aKV}, W^{aGate}], \quad
[kv^b, score^b] = X\,[W^{bKV}, W^{bGate}], \qquad kv,score \in \mathbb{R}^{M \times 2N}
$$

$$
\textbf{else } r = 128 \text{ (C128A):} \qquad
[kv, score] = X\,[W^{KV}, W^{Gate}], \qquad kv,score \in \mathbb{R}^{M \times N}
$$

### 3.2 窗口装配 + ape 偏置

C4A 每 token 行分两段 `[coff0 | coff1]`，第 i 组窗口 = 前一组 4 token 的 coff0 段 ∪ 当前组 4 token 的 coff1 段（overlap，8 行）；C128A 窗口 = 当前组 128 token 整行：

$$
\textbf{if } r = 4:\quad
score_i' = \begin{bmatrix}
score^a_{[4(i-1)+1:4i, :]} \\ score^b_{[4i+1:4(i+1), :]}
\end{bmatrix} + \text{ape}, \qquad kv_i = \begin{bmatrix}
kv^a_{[4(i-1)+1:4i, :]} \\ kv^b_{[4i+1:4(i+1), :]}
\end{bmatrix}, \quad \text{8 行} \times 2N
$$

$$
\textbf{else}:\quad
score_i' = score_{[r(i-1)+1:ri, :]} + \text{ape}, \qquad kv_i = kv_{[r(i-1)+1:ri, :]}, \quad r \text{ 行} \times N
$$

### 3.3 逐通道 softmax（对窗口按列独立归一）

$$
\textbf{任意 } r:\quad
S_i' = \text{softmax}(score_i') \in \mathbb{R}^{r \times (\cdot)},\quad
S_i'[j,k] = \frac{\exp\!\big(score_i'[j,k] - \max_j score_i'[j,k]\big)}{\sum_{j'=1}^{r} \exp\!\big(score_i'[j',k] - \max_j score_i'[j,k]\big)}
$$

### 3.4 加权压缩（Hadamard + 沿压缩轴求和）

$$
\textbf{if } r = 4:\quad
(S_H)_i = S_i' \odot \begin{bmatrix}
kv^a_{[4(i-1)+1:4i,:]} \\ kv^b_{[4i+1:4(i+1),:]}
\end{bmatrix}, \qquad
C_i^{\text{Comp}} = [1]_{1 \times 8}\; (S_H)_i
$$

$$
\textbf{else}:\quad
(S_H)_i = S_i' \odot kv_{[r(i-1)+1:ri,:]}, \qquad
C_i^{\text{Comp}} = [1]_{1 \times r}\; (S_H)_i
$$

### 3.5 后处理（两种配置相同）

$$
\text{RMS}(C^{\text{Comp}}_j) = \sqrt{\frac{1}{N}\sum_{n=1}^{N} \big(C^{\text{Comp}}_{j,n}\big)^2 + \text{eps}}, \qquad
\text{RmsNorm}(C^{\text{Comp}}) = \text{norm\_weight} \cdot \frac{C^{\text{Comp}}}{\text{RMS}(C^{\text{Comp}})}
$$

$$
\text{cmp\_kv} = \text{Rope}\big(\text{RmsNorm}(C^{\text{Comp}})\big) \quad (\text{仅后 } \text{rope\_head\_dim} \text{ 维旋转})
$$

### 3.6 state 递归（两种配置相同）

每个 token 的 `kv`/`score`（或 `kv^a,kv^b`/`score^a,score^b`）按绝对 seq 下标写入分页 `state_cache`；
下个 call 的窗口前驱行（跨 chunk / decode 续算）从 `state_cache` 读。

### 3.7 差异对照

| | C4A（deepseek v4 实测） | C128A |
|---|---|---|
| 压缩比 r / coff c | 4 / 2（overlap） | 128 / 1 |
| 投影 | 双组权重 `W^{aKV},W^{aGate},W^{bKV},W^{bGate}`，输出 `[M, 2N]` | 单组 `[W^{KV},W^{Gate}]`，输出 `[M, N]` |
| 窗口 | 8 行（前组 coff0 ∪ 本组 coff1） | 128 行（本组整行） |
| softmax / 加权 / 求和 | 8 行逐通道，`[1]_{1×8}` | 128 行逐通道，`[1]_{1×128}` |
| 后处理 / state | 相同 | 相同 |

## 4. 优化方案与当前架构

1. **GEMM 拆出**：两个 `MatMulV3` 跑纯 cube，`compressor_epilogue` 只做向量部分；
2. **epilogue 完全串行化**：每核独占完整 head_dim 行并行，各核独立完成窗口→softmax→加权→rms_norm→rope→输出，无跨核同步；
3. **GEMM 预合并**（推荐）：`W_cat = concat([W^KV, W^Gate])` 加载时一次性拼 `[2048,7168]`（不计入运行时），单 `F.linear`（N=2048）→ `chunk` 视图。数学等价，N 翻倍改善 L1 复用。

## 5. NPU graph 模式全矩阵对比（batch 1~8192）

**计时方法**：NPU graph 捕获计算图（`torch.npu.graph`）后 replay 20 次取平均——与 vllm-ascend 生产解码 graph 场景一致，launch 开销被隐藏，测得纯设备侧执行时间。

**数据来源（可复现）**：`benchmarks/ops_profiling/bench_compressor_graph.py`（graph 捕获 + replay 20 次平均，卡 5）→ `msprof_out/compressor_graph_results.csv` → `benchmarks/ops_profiling/analyze_compressor_graph.py`（生成本节全部数值）。

**三种实现**：
- `fused`：单 `compressor` 算子（内部 cube mm + epilogue）
- `split_pack`：权重预合并 `W_cat = concat([W^KV, W^Gate])`（[2048,7168]，不计时），单 `MatMulV3`（N=2048）→ `chunk` → `compressor_epilogue`
- `split_nopack`：两次独立 `MatMulV3`（2×N1024）→ `compressor_epilogue`

### 5.1 decode（B 请求 × 1 token，kv_len=4096）

| B | fused (us) | split_pack (us) | split_nopack (us) | pack 提升 | nopack 提升 |
|---|---|---|---|---|---|
| 1 | 54.2 | **50.2** | 57.9 | 1.08× | 0.94× |
| 2 | 54.7 | **52.5** | 56.5 | 1.04× | 0.97× |
| 4 | 55.5 | **54.0** | 56.8 | 1.03× | 0.98× |
| 8 | 61.3 | **56.7** | 58.0 | 1.08× | 1.06× |
| 16 | 62.6 | 51.0 | **48.4** | 1.23× | 1.29× |
| 32 | 68.2 | **58.9** | 59.4 | 1.16× | 1.15× |
| 64 | 80.5 | **70.6** | 73.7 | 1.14× | 1.09× |
| 128 | 111.5 | **91.7** | 104.2 | 1.22× | 1.07× |

> decode B≤4 时 nopack 反而略慢于 fused（0.94-0.98×，两次 GEMM launch 的碎片化在极小 batch 下无收益）；pack 因单 GEMM 无此问题，B=1 起即胜出（1.04-1.23×）。B≥8 后两者均优于 fused。

### 5.2 prefill（单请求连续 M 行，kv_len=M）

| M | fused (us) | split_pack (us) | split_nopack (us) | pack 提升 | nopack 提升 | pack TFLOPS |
|---|---|---|---|---|---|---|
| 256 | 108.8 | 85.6 | **82.7** | 1.27× | 1.32× | — |
| 512 | 145.3 | 118.9 | **116.2** | 1.22× | 1.25× | 126（51.4%） |
| 1024 | 258.9 | 204.5 | **201.4** | 1.27× | 1.29× | 147（59.8%） |
| 2048 | 499.5 | **332.9** | 346.2 | 1.50× | 1.44× | 181（73.5%） |
| 4096 | 983.1 | **663.6** | 677.0 | 1.48× | 1.45× | 181（73.7%） |
| 8192 | 1992.7 | **1297.5** | 1309.7 | 1.54× | 1.52× | 185（75.4%） |

> prefill 全场景 split 优于 fused，M≥2048 优势放大到 1.44-1.54×（GEMM 占比上升、fused 内部 cube 效率低拖累越明显）。M≤1024 时 nopack 略优（3-7%），M≥2048 时 pack 略优（2-4%）——pack 单 GEMM 的 L1 复用收益在 GEMM 占比高时才显现。

### 5.3 pack vs nopack 单独对比

| 场景 | split_pack | split_nopack | pack 优势 |
|---|---|---|---|
| decode B=1 | **50.2** | 57.9 | 13% |
| decode B=16 | 51.0 | **48.4** | -5%（nopack） |
| decode B=128 | **91.7** | 104.2 | 12% |
| prefill M=256 | 85.6 | **82.7** | -3%（nopack） |
| prefill M=1024 | 204.5 | **201.4** | -2%（nopack） |
| prefill M=8192 | **1297.5** | 1309.7 | 1% |

> pack 与 nopack 整体差距 <5%（decode 极端 B 下 pack 省一次 GEMM launch 收益 ~13%）；`chunk` 是零拷贝视图，无额外开销。pack 在大部分场景略优，nopack 在 GEMM 占比中等时略优。

## 6. 数值正确性（test_compressor_split.py，多轮稳定）

| 用例 | 结果 |
|---|---|
| prefill B=1 q=8192 / B=4 q=512 | rel>1% ≈ 11.5%（bf16 mm 固有精度差，实测上限 10.8-11.4%） |
| decode B=8 / B=64 | rel>1% ≈ 0.6-0.9%，max_abs ≤ 0.125 |
| prefill chunk（startPos 非 4 对齐） | rel>1% ≈ 11.4%，state 全对齐 |
| decode 2-step（state 连续性） | state_max_abs ≈ 0.03，6/6 通过 |

## 8. epilogue 集成后性能验证（2026-08-20）

**测试环境**：Ascend910B4-1，NPU 0（空闲独占），CANN 9.0.0
**代码状态**：epilogue kernel 支持 bf16/fp16 norm_weight/rope（模板参数 T_NORM/T_ROPE + cast fp32），def.cpp 恢复 config910 对齐 fused dtype 组合；生产路径实测 rope cache 恒 fp32、norm_weight bf16 → 走 `8f8aba1e`（bf16/bf16/fp32）变体
**测量方法**：与 §5 一致（graph 捕获 + replay 20 次平均）；epilogue 单 kernel 用 msprof op（Task Duration + PipeUtilization）
**脚本**：`bench_compressor_graph.py`、`bench_msprof_compressor.py`、`bench_compressor_kernel_time.py`

### 8.1 graph 全矩阵（本次 vs 2026-08-07 报告）

**decode（B 请求 × 1 token）**：

| B | fused | pack 本次 | pack 旧 | nopack 本次 | 变化(pack) | f/pack |
|---|---|---|---|---|---|---|
| 1 | 56.7 | **44.0** | 50.2 | 50.8 | -12% | 1.29× |
| 4 | 59.3 | **47.9** | 54.0 | 51.3 | -11% | 1.24× |
| 8 | 69.0 | **51.0** | 56.7 | 53.0 | -10% | 1.35× |
| 16 | 66.1 | **47.2** | 51.0 | 46.0 | -7% | 1.40× |
| 32 | 73.0 | **50.4** | 58.9 | 50.1 | -14% | 1.45× |
| 64 | 81.9 | **55.2** | 70.6 | 58.6 | -22% | 1.48× |
| 128 | 108.2 | **66.5** | 91.7 | 85.7 | -27% | 1.63× |

> decode 全场景 split_pack 显著改善（-7%~-27%，B 越大收益越明显）；B=16 起 nopack 反超 pack（N=1024 双 GEMM 在 GEMM 占比低时碎片化收益不抵 pack 单 GEMM 的 L1 复用）。

**prefill（单请求连续 M 行）**：

| M | fused | pack 本次 | pack 旧 | nopack 本次 | 变化(pack) | f/pack |
|---|---|---|---|---|---|---|
| 256 | 111.6 | 84.9 | 85.6 | **83.7** | -1% | 1.31× |
| 512 | 145.4 | 122.0 | 118.9 | **120.3** | +3% | 1.19× |
| 1024 | 261.3 | 202.8 | 204.5 | **202.2** | -1% | 1.29× |
| 2048 | 510.5 | **329.3** | 332.9 | 346.0 | -1% | 1.55× |
| 4096 | 988.7 | **650.1** | 663.6 | 659.6 | -2% | 1.52× |
| 8192 | 1996.5 | **1256.8** | 1297.5 | 1267.2 | -3% | 1.59× |

> prefill 与报告基本持平（±3%，卡间/温度抖动）；M≥2048 时 pack 优势稳定（1.52-1.59× vs fused）。

### 8.2 epilogue 单 kernel 深度分析（msprof op，prefill M=8192）

| 指标 | 数值 | 诊断 |
|---|---|---|
| Task Duration | **96.5us**（报告时 ~160us，-40%） | bf16 cast 路径 + 双缓冲优化生效 |
| Block Dim | 40（20 AIC × 2 复用，纯 AIV） | 40 个 AIV 全用上 |
| **aiv_vec_ratio** | **0.73** | **vector 计算是瓶颈**（softmax/rmsnorm/rope/加权） |
| aiv_mte2_ratio | 0.31 | GM→UB 读仅 1/3 占用 |
| aiv_mte3_ratio | 0.33 | UB→GM 写仅 1/3 占用 |
| main_mem 读写带宽 | 读 9.0 + 写 17.6 GB/s/核 → 总 ~1.07 TB/s | ~67% HBM SOL |

**SOL 对照**：epilogue M=8192 流量 ≈ 67MB（mm 读 33.5 + state 写 16.8 + state 读 8.4 + 输出 8.4）→ mem SOL ≈ **42us**；实测 96.5us = **2.3× mem SOL**——瓶颈不在访存（mte2/mte3 仅 ~0.3），而在 **vector 指令数**（每压缩行：Cast×2 → AddApe → 逐列 softmax(exp) → 加权求和 → RmsNorm(平方/rsqrt) → RoPE → Cast）。vec_ratio 0.73 仍有 27% 空闲（sync/MTE 等待），理论可达 ~70us（1.7× mem SOL）。

### 8.4 C128A 性能验证（ratio=128, coff=1, state_block=32）

**graph 全矩阵**（`bench_compressor_graph.py --ratio 128`）：

| 场景 | fused | pack | nopack | f/pack | 场景 | fused | pack | nopack | f/pack |
|---|---|---|---|---|---|---|---|---|---|
| decode B=1 | 43.8 | **39.3** | 40.2 | 1.11× | prefill M=256 | 71.0 | **61.8** | 73.6 | 1.15× |
| decode B=2 | **47.3** | 48.0 | 46.4 | 0.99× | prefill M=512 | 114.1 | **80.2** | 85.0 | 1.42× |
| decode B=4 | **48.3** | 53.6 | 50.9 | 0.90× | prefill M=1024 | 137.7 | 127.1 | **121.4** | 1.08× |
| decode B=8 | **54.1** | 57.8 | 54.8 | 0.94× | prefill M=2048 | 225.8 | **200.1** | 199.6 | 1.13× |
| decode B=16 | **58.7** | 59.0 | 59.6 | 0.99× | prefill M=4096 | 408.7 | **340.8** | 338.2 | 1.20× |
| decode B=32 | **76.8** | 79.5 | 83.3 | 0.97× | prefill M=8192 | 876.3 | **673.4** | 713.1 | 1.30× |
| decode B=64 | **109.6** | 134.9 | 140.5 | 0.81× | | | | | |
| decode B=128 | **168.0** | 300.1 | 302.8 | 0.56× | | | | | |

> **与 C4 的关键差异**：C128 decode 大 B（B≥64）在**满组测试场景**（全部请求同位置、同时组满）下 fused 反而快（B=128: 0.56×）——这是**上界场景的假象**（128 请求同时 produce，非生产形态）；**生产形态（请求位置分散、1/128 产出率）下 split 全面胜出**（见 8.4.1）。msprof app 确认满组 B=128：fused 161.4us vs split1 272.3us（0.59×）。

**msprof op pipeline 对比（decode B=128）**：

| 指标 | split epilogue | fused Compressor |
|---|---|---|
| Task Duration | 255.9us | **169.7us** |
| Block Dim | 40（纯 AIV） | **20（mix，AIC+AIV 协作）** |
| aiv_vec_ratio | 0.16 | 0.24 |
| aiv_mte2/mte3 | 0.08 / 0.03 | 0.20 / 0.07 |
| **aiv_scalar_ratio** | **0.73（瓶颈）** | 0.45 |
| aic_cube_ratio | — | 0.05（GEMM 极小） |

> **根因（满组上界）**：C128 decode B=128 只有 1 个完整组（128 行 × 512 列）→ 40 核分列后每核仅 ~1638 元素，任务粒度太小，**scalar 控制流/地址计算开销主导（0.73）**；fused 用 20 核 mix 模式减少核间调度开销（scalar 0.45）。C4 无此问题（B=128 → 16 组 × 8 行 × 1024 列，每核工作量 ~2 倍，vec_ratio 0.73 计算 bound）。

### 8.4.1 生产形态验证（1/128 产出率，请求位置分散）

满组场景（全 produce）是**性能上界**；生产 decode 请求位置分散，组满概率 ~1/128（`g.produce = gStart+r <= P+S`），**绝大多数任务只做 mm load + Cast + SaveState（无 127 行历史读、无 softmax）**。

**decode 链（256 步，P 递增，state 真实累积，每 128 步 1 次 produce = 2/256 步）**：

| 场景 | fused 平均 | split1 平均 | produce 步 | 非 produce 步（254/256） | f/s1 |
|---|---|---|---|---|---|
| 链 B=8 | 37.4 | **23.2** | 45.4/43.9 | 37.4 / 23.2 | **1.61×** |
| 链 B=32 | 50.6 | **34.7** | 74.4/73.1 | 50.6 / 34.7 | **1.46×** |

**分散单步（128 token、恰 1 个 produce）**：

| 场景 | fused | split1 | f/s1 |
|---|---|---|---|
| 分散 B=8 | 53.7 | **33.4** | 1.61× |
| 分散 B=32 | 61.0 | **31.9** | 1.91× |
| 分散 B=128 | 97.8 | **48.8** | **2.01×** |

**满组 vs 生产（B=128）**：

| | fused | split1 |
|---|---|---|
| 满组全 produce（上界） | 168.0 | 300.1 |
| 分散 1 produce（生产） | 97.8 | **48.8** |
| 下降 | -42% | **-84%** |

> **关键结论**：① 生产形态下 **C128 split 全场景胜出 1.46-2.01×**，满组"fused 反超"（0.56×）确认为测试场景假象；② 不 produce 的任务只做 mm+Cast+SaveState → split1 降 84%，而 fused 仅降 42%——**fused 的非 produce 步也做窗口装配（37.4 vs 23.2，+61%），是 fused 生产形态慢的根因**；③ produce 步额外开销（127 行历史读 + softmax）：fused +8us、split1 +14us。

### 8.5 结论

1. **epilogue dtype 适配（bf16/fp16 cast）不引入性能回退**：decode 反而显著改善（B≥8：-10%~-27%，B=128 最明显 91.7→66.5us），prefill 持平（±3%）。
2. **split_pack 全场景优于 fused**（生产形态，1/128 产出率）：prefill 1.19-1.59×、decode 链 1.46-1.61×、分散单步 1.61-2.01×；满组上界场景（全 produce）不具代表性（C128 大 B 的"fused 反超"是假象）。
3. **fused 的生产形态短板**：非 produce 步也做窗口装配（C128 decode 链 B=8: 37.4 vs split1 23.2，+61%）；split 路径天然只在 produce 时读历史/softmax。
4. **已实施优化（无回归，c128 全量回归 PASS）**：Phase2NormRope 批量（NORM_BATCH=4，对齐 C4 FlushNormRope）；UB 统一池管理（单 TBuf + GetWithOffset，178.5KB < 192KB，阶段一/二按生命周期复用）。满组上界的 255us 中 padding Locate 浪费（tiling 上界 2B）与非 produce 任务固定开销占大头——生产形态下已不是瓶颈。
5. **生产建议 split_pack 路径**：权重预合并单 GEMM + chunk + epilogue，全场景（prefill + decode 生产形态）无劣化。

## 9. 附录：历史结论


1. **动机成立**：fused 的 GEMM 因 N=1024 太小、A 矩阵 L1 复用差，cube 利用率仅 ~50%，prefill M=8192 耗时 1992.7us。
2. **graph 全矩阵（batch 1~8192）结论**：
   - **prefill 全场景 split 大幅优于 fused**：M≥2048 提升 1.44-1.54×（M=8192：1992.7→1297.5us，75.4% SOL）；M=256-1024 提升 1.22-1.32×；
   - **decode 大部分场景 split 优于 fused**：B≥8 提升 1.06-1.29×；B≤4 时仅 pack 胜出（1.03-1.08×），nopack 因两次 GEMM 碎片化略慢（0.94-0.98×）；
   - **pack vs nopack 整体差距 <5%**：pack 在 GEMM 占比高的场景（decode 大 B、prefill M≥2048）略优；nopack 在中等 M 略优；`chunk` 零拷贝无额外开销。
3. **建议实施 split_pack**：权重预合并单 GEMM + chunk 视图，全场景（含 decode 小 B）无劣化场景，且仅一次 GEMM launch。
4. **剩余空间**：prefill M=8192 split_pack 1297.5us 中 epilogue ~160us 相对内存下限 ~66us 仍有 2.4×（向量指令效率、rope、ape 常驻等）；C128A 流程同构可复用优化。
