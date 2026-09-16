# r31 直接分页 consumer：性能未达门槛

r31已移除连续K workspace。18项native/graph正确性通过，但本轮完整selector
性能未达到主要prefill至少1.2×、非目标退化不超过3%的验收要求，不能称为性能完成。
本报告只测已提交的直接分页r31，不复跑已撤回的连续K实现。

## 完整selector结果

同一组seed41合成Q/K经过真实source kernel生成候选；source耗时不计入consumer。
这些不是模型checkpoint trace。每shape三路独立oracle通过，3轮×12样本、graph
unroll4、交替测量顺序，完整原始样本和包/库指纹保留在artifacts/qli-fused/r31。
表中原先简称dense的对照准确名称为 **full-scan + candidate mask**：
`selector_call(legacy_offset=offsets)`的`offsets`是全零Tensor，仍传
`candidate_mode=2`及同一份candidate。提供offset Tensor使dispatch选择generic
路径，零值不改变位置；它计算整个context的QK，然后在top-k前通过候选
mask压低集合外score，**不是unrestricted top-k**。原始JSON中的`native`键
对应此路径，保留原始证据不改名。mode2 paged是会排序候选的旧分页融合路径，
mode4保留source候选顺序，二者DMA发射序列不同。单位µs：

| Shape | r31 median / P95 | full-scan + candidate mask median | mode2 paged median | 对full-scan + mask加速 |
| --- | --- | --- | --- | --- |
| b4-t128-n32771 | 1610.89 / 1628.24 | 1539.42 | 1430.87 | 0.956x |
| b8-t512-n32771 | 5625.98 / 5649.38 | 3445.17 | 4343.09 | 0.612x |
| t128-n131075 | 1610.75 / 1614.59 | 2411.22 | 1500.36 | 1.497x |
| t128-n32771 | 1601.58 / 1617.05 | 1091.05 | 1422.93 | 0.681x |
| t128-n4097 | 687.73 / 693.96 | 362.32 | 635.31 | 0.527x |
| t512-n131075 | 5750.62 / 5755.04 | 6793.55 | 4797.19 | 1.181x |
| t512-n32771 | 5719.40 / 5747.43 | 3442.66 | 4356.10 | 0.602x |
| t512-n4097 | 2182.22 / 2190.99 | 917.80 | 1676.18 | 0.421x |

r31固定用户workspace为6,268,160B（20核组双slot），K workspace为0。
实测T128增量allocated峰值约22.48MiB，T512约24.95–24.97MiB，
含CANN固定16MiB和完整selector输出/后处理，不能全部称为K内存。

## Oracle口径与旧路径边界

三路输出分别通过同一个受限候选集合top-k oracle：逐query应用因果长度与物理页
合法性，检查输出数量、去重、候选集合成员及分数边界。oracle要求确定高于cutoff
的项全部入选，允许cutoff并列项和有界FP32归约误差造成的边界替换；不要求不同
实现的ID张量bitwise相同。本轮没有保存逐实现逐ID强制一致的对照结论。

full-scan + candidate mask旧实现只把候选外score压到有限NEG_HUGE，保留其
位置ID。合法候选不足512时，它可能以集合外位置补足top-k，无法满足r31的-1
填充合同。本次8shape没有触发该限制：对保存candidate逐行CPU枚举，T512/4K
最少3586个合法位置、T128/4K最少3970，其余shape至少16377；所有物理页有效，
不足512的行数均为0。这只支持本轮fixture上的受限top-k比较，不能把旧路径当作
任意候选输入的通用等价fallback。

枚举证据：`artifacts/qli-fused/r31/candidate-cardinality-audit.json`，SHA为
`e71e13de256e8e6c88a5fa76a829542c1c2f4c174c66df883f162e172a7091d8`。

## 新路径全核profile

T128/N32771，20个AIC与40个AIV全部统计；8个核组处理7query、12组处理6query。
kernel wall=1514.30µs。以下为每核在整个kernel内的累计
counter mean，不是单query时延；计数可以重叠，不能相加当作wall。此次未触发总
counter时间异常检查；原始CSV/SHA和每核min/mean/max全部保留。

| 角色/计数 | 累计mean µs |
| --- | --- |
| AIC Cube | 61.83 |
| AIC MTE1 | 120.71 |
| AIC MTE2 | 743.15 |
| AIC FixPipe | 725.82 |
| AIC Scalar | 539.53 |
| AIC metadata-ready等待 | 505.34 |
| AIC slot ACK等待 | 105.11 |
| metadata AIV MTE2 | 557.10 |
| metadata AIV Scalar | 556.56 |
| top-k AIV Vector | 272.72 |

Cube active/wall约4.08%。源码确认每query分页K直接GM→L1，小块ND2NZ以
block8（1024B）发射；metadata AIV只读candidate/page/scale，没有K搬运。
同query仍需metadata-ready后才启动Cube；不同query可通过双slot重叠。

## 可确认的限制

对本次T128冻结candidate做CPU重放，按照源码分页K读取的相邻同页合并
规则，mode4平均2047.70次K ND2NZ/query，mode2排序对照1325.79次/query。
计数是源码和输入推导，不是硬件DMA/HBM计数；完整数据见
direct-paged-dma-call-analysis.json。它说明去掉候选排序后不能期待与旧mode2
相同的分页小搬运发射数；没有据此建议恢复排序或K staging。

本轮确认需要研究直接paged的小块ND2NZ与Scalar发射、metadata准备及重叠；
尚未通过指令timeline分离这些开销，不能仅凭MTE2/FixPipe累计counter认定唯一
瓶颈，也不能用旧连续路径527µs外推。没有运行新调优、修改tiling或降低门槛。

## 证据指纹

包SHA：`2d4ea5ac1618c6642c25025dec2730001f682e08b39b0a575c14773a20bd2f4b`。
完整8shape JSON、summary和原始CSV的路径/SHA列于
`artifacts/qli-fused/r31/performance-evidence-manifest.json`，该清单SHA为
`f686e6ccc5c82c6a6b401e48a908b0947a0bd3888c0230b5ef45b90f1a06a8b0`。
全核profile summary：`profile-t128-n32771-paged-unique-source-r1-summary.json`，SHA为
`6ccb5c4cc14f8281e17ad86e85467a5749b7c7091745b388aba2e4241b7d7eb3`。
原始`PipeUtilization.csv` SHA为
`5dca3bb2b454ee8ec07c152b36fcb3035614867dbfcd8c90b20d378bb03af367`；
`OpBasicInfo.csv` SHA为
`0f40a8de0ebca083d84c6b370ae8d3f71b769a3fded313d550ae5f512e386d0a`。
