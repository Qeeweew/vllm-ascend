# V4.1 trusted unique consumer：Cube 直接分页读 K

2026-09-16：原 Vector 搬 K 到连续 GM workspace 的方案已撤回，入口和实现删除。
mode4 的 `trusted_unique_candidates=True` 仍表示输入中有效 candidate block ID
唯一；它不承诺候选顺序、相邻 query 共享 K、没有内部空洞或满 512 个合法输出。

每个核组的偶数 AIV 只解析 candidate/page 合法性，准备物理 K offset、逻辑位置、
key scale 和展开后的 weight×query scale。Cube 通过现有 N128 paged ND2NZ 路径
直接读取原始 K cache 到 L1，完成 INT8 QK、FixPipe ReLU、第二次 matmul 的
weighted head reduction。奇数 AIV 读取缩放后所需数据，对完整 query 的候选做
scale/mask/top-k512。没有候选排序或去重，也没有跨 query K 复用。

物理 offset 不是 K 内容。现有 paged Cube DMA 接口需要 offset；AIV 还必须依据
同一合法性规则取得 key scale 和输出位置。先共用这次解析，避免 AIC/AIV 重复
维护页表边界与因果掩码逻辑。metadata-ready 只发布这份小记录，不表示 K 已搬运。
所有 K DMA 的目的地都是 Cube L1；AIV 没有 K tensor 或 K UB buffer。

每个 record 为 156704 bytes：weight 1024、offset 16384、position 8192、
padded scale 65536、reduced score 65536、state32。每核组两个 record，以完成
metadata/Cube/top-k 的 ownership 交接；20 组共 6268160 bytes，约 5.978MiB。
它不随 query 数增长，另加 CANN 固定 workspace。删除的连续 K 方案每槽还额外
分配 2097152 bytes K。满16K候选时，新路径每 query 的逻辑 K GM读取为2MiB，
没有旧路径的额外2MiB GM写回及再次读取；这不是对实际HBM流量或延迟的预测。

首版采用已验证的原 N128 Cube 几何，撤掉 N256/N512 和 Q/W常驻实验。
metadata 与上一个 query 的 Cube 运算可在两个 record 间重叠；record 只有在
metadata producer 与完整 top-k owner 都确认后才复用。小 token 数暂用split1。
不改变非法页、future位置、空请求、graph padding、负score、不足512补-1合同。

r31已完成完整构建及源码/安装指纹核验，CPU协议10项（1280随机调度）通过，
manual hooks通过。包SHA：
`2d4ea5ac1618c6642c25025dec2730001f682e08b39b0a575c14773a20bd2f4b`。
静态审计在`artifacts/qli-fused/r31/paged-source-workspace-audit.json`，确认无GM K写回。
18项native测试已通过（64.37s）：T1–1024 prefill、多batch decode B1/8/32/64、
混合多batch prefill、真实source唯一性→consumer、负score/不足512，以及bucket
128/2048各16次变化输入graph replay。测试日志在r31/native-full.log。
完整selector性能及全模型集成尚未验证；旧连续路径的READY527µs不能外推。
后续profile同时列kernel wall、每核query数和各核累计counter，避免把累计等待
当成单query阶段时延。性能门槛仍为主要prefill至少1.2×且非目标退化不超过3%。

AIC在metadata-ready之后、读取复用record之前，使用CANN的
`CacheLine::ENTIRE_DATA_CACHE` / `DcciDst::CACHELINE_OUT` 失效标量GM cache，
使state与所有candidate offset属于本query代次。仅失效state一条cache line不能
保护循环复用的offset表；r30构建因审计发现这一点被r31替代，未进入设备测试。
