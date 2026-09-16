# 已撤回：连续 K workspace 方案（历史证据）

**2026-09-16：按用户要求撤回，不再保留可调用的连续 K 实现。**
r23–r29 数据仅用于记录失败实验，不代表当前 mode4，也不用于推导直接分页路径性能。
当前 mode4 只允许 Cube 从原分页 cache 直接搬 K 到 L1；设计见
[PAGED_UNIQUE_DESIGN.md](PAGED_UNIQUE_DESIGN.md)。以下内容均为撤回前的历史记录。

状态：r23 完整隔离包及15项真实 NPU 测试通过；六场景性能未通过验收。
实测与后续工作见 [r23结果](CONTIGUOUS_GATHER_RESULT.md)。
本实现遵循用户最新分工：Vector gather，Cube 完成 QK、ReLU 和第二次矩阵
乘法的 head 加权归约，Vector 只做最终 key scale、mask 和完整 query top-k。

## 数值与唯一性合同

新增 `candidate_mode=4`，仅允许来自 source 的唯一有效 block IDs；原 mode2
的任意顺序、重复候选集合语义不变。mode3 已用于 off。重复负 sentinel 允许，
正的有效 block ID 必须唯一。mode4 不排序、不去重，按输入候选顺序填连续 K。

真实 source `ProcessCandBlockTopk` 对每个不相交 S2 tile 构造
`tileBlockBase+j`，block top-k merge 只选取这些不同位置；pin 最新 block
仅修改已有分数，不插入第二份 ID。因此有效 source IDs 天然唯一，但按
score 排列，并非逻辑位置升序。测试必须检查真实 source 输出再调用 consumer。
旧测试 `candidates_for(adversarial=False)` 也可能因覆盖 slot0 为最新 block
而产生重复，不能把那个输入直接称作 mode4 的合法合同。

数值路径保持原 native rounding：

- INT8 QK 经 FixPipe 除以 1024、ReLU，并舍入 FP16，结果留在 L1。
- `FP16(weights * query_scale)` 送第二次 Cube WS，输出 FP32 每候选分数。
- Vector 乘 key scale、屏蔽无效位置，在该 query 全部 16384 个位置中选512。

少于512个有效位置以 -1 补齐；负分数有效，填充使用负无穷而非0。候选屏蔽
包含负 ID、越界 ID、非法物理页、因果未来位置、空请求、graph padding。
mode4 不支持的静态形状在 host 明确拒绝，不转入存在候选外填充问题的旧分支。

Python 显式接口：

```python
AscendIndexerV41Ops(1, "consumer", trusted_unique_candidates=True)
```

该参数默认False；只允许 native CR1 consumer，不能与旧 B1 workspace 组合。
没有 host tensor value 读取、环境开关或自动识别候选唯一性。

## 数据流与资源

每个物理核组有固定两槽。偶数 AIV gather，AIC 读取连续 INT8 K 做两次
matmul，奇数 AIV 对整个 query 做 top-k。prefill 按 T 分配 query，mode4
当前只做 split1；小 T 分 N 的2/4份方案仍待单独实验。不得把本轮当作 decode
性能验收。

每槽2254336 bytes：156704 bytes 的原 score/scale/position record、480B对齐padding，加
2097152 bytes 的连续 INT8 `[16384,128]` K。为复用已验证 top-k epilogue，
原 record 中16KiB offset 区暂保留但不写不读。scale按block8填充到32B，
仍占64KiB；后续可独立压缩，此轮不混入数值或向量归约变化。

20核总用户 workspace90173440 bytes，约85.996 MiB，不随 T 增长。CANN 固定
workspace另计。gather UB145056 bytes，topk UB71840 bytes。Cube继承旧
QK/WS service，L1实际384KiB、L0A/B各64KiB、L0C128KiB；H32紧凑N256布局
尚未实施，不宣称这些资源已优化。

Vector每次通过两个64KiB UB缓冲之一搬64个block8，写一次连续64KiB K GM；
另一缓冲可并行搬运。K源页查表后按1KiB块读取，scale按16B读取并填充32B。
AIC每个N128 tile只发一次连续ND→NZ DMA，完全去掉原来的GM地址表读取和
候选分页ND→NZ小搬运循环。尾部没有有效候选的N128 tile不做QK；输入内部
空洞维持位置不变，由epilogue屏蔽。

这里只保证 **Cube输入的连续性**。源cache gather仍有离散块与小scale搬运，
还增加了一次K GM写回/读取；是否更快必须完整selector与msprof验证。

## 双槽同步协议

MODE2 的 AIV→AIC 为同核组两AIV聚合，AIC→AIV 为双播。

|角色|操作|
|---|---|
|gather AIV|复用i-2前等FREE；写K/scale/position/state；发READY和ACK的producer半；延后一代消费SCORED|
|topk AIV|初始为两个槽预发READY credit；等SCORED；完整topk后发ACK；有后续i+2才补READY credit|
|AIC|复用i-2前等ACK并发FREE；等READY；QK+WS；用PIPE_FIX发SCORED；末尾drain最后1/2个ACK|

FREE在topk完成且该槽Cube K消费结束后才能产生，因此score和K均不会被提前
覆盖。gather(i+1)、Cube(i)、topk(i-1)可以并行。K起点和每槽stride按512B对齐。没有全query Prepare、全核
SyncAll或全局READY barrier。state通过标量读取前DCCI失效对应cacheline；
DMA/FixPipe数据可见性由其真实producer pipe发flag建立。

`test_indexer_v41_contiguous_protocol.py` 1280个随机调度覆盖0/1/2/3/4/5/
16/33/129/513 query，检查slot generation、读写占用、全部MODE2 token drain。
这是抽象协议检查，不是硬件visibility或graph正确性的证明。

## 验证与性能门槛

- 真实source唯一性→mode4 consumer，独立CPU oracle。
- B1、多batch decode、T32/64/128/512 prefill，后续扩展T1024与更大bucket。
- ragged、空请求、坏页、gapped cache、负分数、不足512、全无效。
- T128 graph bucket，16次修改Q/K/scales/weights/长度/分页/候选，覆盖每槽多代复用。
- `benchmark_indexer_v41_prefill.py --candidate-mode 4` 对同一批唯一输入测新
  mode4、mode2 paged fused、mode2 dense legacy；完整selector包含相同后处理。
- T128/512 ×4K/32K/128K先沿用完整shape矩阵；主要T≥128目标至少1.2x，
  非目标回退≤3%，不允许用长N优势掩盖短N退化。
- msprof op全20个AIC/40个AIV的Cube/MTE/FixPipe/Vector/scalar/flag等待分布，
  同时报告wall time、有效QK FLOPs和执行工作量。不用core0代替全卡。
  Cube比例仅作诊断；不同query候选K不复用，以完整selector延迟/吞吐/内存为目标，
  不要求达到dense/H64利用率，不为提高比例增加无用预处理或算术。

完整构建、source/object/package SHA、native/graph和性能结果未全部通过前，
禁止默认启用或以降低门槛宣布完成。
