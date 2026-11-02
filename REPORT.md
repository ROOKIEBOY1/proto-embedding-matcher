# BGE-small-en-v1.5 Proto 语义匹配实测报告

实验日期：2026-11-02（Asia/Hong_Kong）。

## 范围与实验完整性

本实验使用合成相机 Proto；没有真实产品协议或生产查询。所有标准答案在推理前冻结，未增训模型。184 条已支持测试样本来自 46 个成员 × 4 个固定模板；另有 40 条人工编写的合成未知样本。开发集为 92 条已支持 + 20 条未知，共 112 条；测试集 224 条。模板来源逐行记录，不能视为 184 个独立业务意图，也没有声称经过用户或第三方人工审核。

开发集和测试集查询不重叠；正例共享候选标签但模板不同，属于域内表达留出。未知数值与负例表达分离。阈值只按开发集已知接受率与未知拒绝率的平均值标定，各 A/B/C 与全局/层级配置独立标定，同分取较高阈值；测试集从不参与选择。

## 部署与可复现信息

- 模型：BAAI/bge-small-en-v1.5；revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`。
- 官方直连失败，使用 hf-mirror.com 的固定 revision；权重和数据 SHA256 见 results/environment.json。
- Python 3.12.14；设备 cuda，NVIDIA GeForce RTX 3060 Laptop GPU；PyTorch 2.5.1+cu121；CUDA build 12.1。
- Sentence Transformers 3.4.1；Transformers 4.48.3。
- 33,360,000 参数；384 维；最大长度 512 tokens。
- 完全本地 safetensors 加载，local_files_only=True；HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1。
- 短对称参数匹配不加 retrieval instruction；A/B/C 共用同一查询向量。L2 归一化 float32 点积即余弦相似度；分数不是概率。

## 编码策略与规则

- A：原始 Proto 成员名；字段层使用原始 enum 类型名。
- B：去公共 enum 前缀、下划线/CamelCase/数字到单位拆分，保留 Type；不拆复合词。
- C：在 B 上移除字段 Type；slowmotion→Slow Motion，videonormal→Normal Video，timelapse→Time Lapse；daylight 保留，mjpeg 保留大写；MINUS_/PLUS_/ZERO_ 规范成有符号物理数值。
- C 没有加入用户同义表达、答案查表或 acutance→sharpness、PLUS→HIGH 等规则；上述规则是人工先验，C 的收益不能全部归因于 Embedding 泛化。
- 帧率、分辨率、ISO、曝光数值来自名称，绝不使用 wire ID 作物理值；不同 enum 即使 wire ID 相同也视为不同答案。

## 独立测试集总表

Top1 为已支持样本的最终正确接受率（已知误拒也计错）；Top3 为纯排名召回。字段准确率统计已支持样本。FAR 为 40 条未知被接受的比例，未知拒绝率=1−FAR。numeric 是额外规则对照，不能计入纯 Embedding。

|编码|检索|决策|Top1|Top3|字段|FAR|未知拒绝|已知误拒|P50 ms|P95 ms|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A|global|forced|85.33%|96.74%|97.83%|100.00%|0.00%|0.00%|11.073|16.200|
|A|global|threshold|59.24%|96.74%|97.83%|40.00%|60.00%|38.04%|11.072|16.199|
|A|global|numeric|90.22%|96.74%|97.83%|37.50%|62.50%|1.63%|11.101|16.224|
|A|global|combined|60.87%|96.74%|97.83%|5.00%|95.00%|38.59%|11.097|16.216|
|A|hierarchical|forced|82.61%|94.02%|95.65%|100.00%|0.00%|0.00%|22.275|29.290|
|A|hierarchical|threshold|59.24%|94.02%|95.65%|40.00%|60.00%|38.04%|22.275|29.290|
|A|hierarchical|numeric|87.50%|94.02%|95.65%|37.50%|62.50%|4.35%|22.293|29.312|
|A|hierarchical|combined|60.87%|94.02%|95.65%|5.00%|95.00%|38.59%|22.292|29.311|
|B|global|forced|86.41%|95.65%|96.20%|100.00%|0.00%|0.00%|11.072|16.198|
|B|global|threshold|67.93%|95.65%|96.20%|40.00%|60.00%|29.89%|11.072|16.197|
|B|global|numeric|89.67%|95.65%|96.20%|37.50%|62.50%|0.54%|11.096|16.214|
|B|global|combined|69.02%|95.65%|96.20%|7.50%|92.50%|30.43%|11.097|16.213|
|B|hierarchical|forced|84.24%|92.93%|95.65%|100.00%|0.00%|0.00%|22.276|29.285|
|B|hierarchical|threshold|67.93%|92.93%|95.65%|40.00%|60.00%|29.89%|22.275|29.285|
|B|hierarchical|numeric|87.50%|92.93%|95.65%|37.50%|62.50%|4.35%|22.293|29.308|
|B|hierarchical|combined|69.02%|92.93%|95.65%|7.50%|92.50%|30.43%|22.292|29.304|
|C|global|forced|88.04%|95.65%|96.74%|100.00%|0.00%|0.00%|11.080|16.200|
|C|global|threshold|70.11%|95.65%|96.74%|42.50%|57.50%|28.26%|11.079|16.199|
|C|global|numeric|90.76%|95.65%|96.74%|37.50%|62.50%|1.09%|11.112|16.221|
|C|global|combined|71.74%|95.65%|96.74%|5.00%|95.00%|28.26%|11.099|16.216|
|C|hierarchical|forced|85.87%|92.93%|95.65%|100.00%|0.00%|0.00%|22.276|29.285|
|C|hierarchical|threshold|70.11%|92.93%|95.65%|42.50%|57.50%|28.26%|22.276|29.285|
|C|hierarchical|numeric|88.59%|92.93%|95.65%|37.50%|62.50%|4.35%|22.309|29.301|
|C|hierarchical|combined|71.74%|92.93%|95.65%|5.00%|95.00%|28.26%|22.293|29.301|

仅由 112 条开发集样本标定的余弦阈值：

|编码|全局阈值|两级阈值|
|---|---:|---:|
|A|0.844926|0.844926|
|B|0.861454|0.861454|
|C|0.876487|0.876487|

forced=纯 Embedding Top1 强制；threshold=开发集阈值；numeric=数值全候选精确校验，非数值无法由该规则判定未知；combined=数值校验+阈值（CLI --guard）。

层级第一级仅编码输入字段，选 enum；第二级使用完整 field:value 查询在该 enum 内排名。全局字段预测为 Top1 所属 enum。字段层误判会封死正确成员；数值 guard 也只验证模型预测的 enum，不能证明字段选择正确。

## 144 FPS 关键证据

|编码|检索|Top1|余弦|Top2|余弦|Top3|余弦|Top1-2差|
|---|---|---|---:|---|---:|---|---:|---:|
|A|global|VIDEO_FRAME_RATE_TYPE__1440FPS|0.905964|VIDEO_FRAME_RATE_TYPE__240FPS|0.828721|VIDEO_FRAME_RATE_TYPE__24FPS|0.827050|0.077243|
|A|hierarchical|VIDEO_FRAME_RATE_TYPE__1440FPS|0.905964|VIDEO_FRAME_RATE_TYPE__240FPS|0.828721|VIDEO_FRAME_RATE_TYPE__24FPS|0.827050|0.077243|
|B|global|VIDEO_FRAME_RATE_TYPE__1440FPS|0.940210|VIDEO_FRAME_RATE_TYPE__240FPS|0.856550|VIDEO_FRAME_RATE_TYPE__24FPS|0.853773|0.083659|
|B|hierarchical|VIDEO_FRAME_RATE_TYPE__1440FPS|0.940210|VIDEO_FRAME_RATE_TYPE__240FPS|0.856550|VIDEO_FRAME_RATE_TYPE__24FPS|0.853773|0.083659|
|C|global|VIDEO_FRAME_RATE_TYPE__1440FPS|0.957961|VIDEO_FRAME_RATE_TYPE__24FPS|0.873929|VIDEO_FRAME_RATE_TYPE__240FPS|0.871807|0.084032|
|C|hierarchical|VIDEO_FRAME_RATE_TYPE__1440FPS|0.957961|VIDEO_FRAME_RATE_TYPE__24FPS|0.873929|VIDEO_FRAME_RATE_TYPE__240FPS|0.871807|0.084032|

即使分数很高或 margin 较大，也不代表数值相等。数值 guard 在预测 enum 的全部成员中寻找规范化数值完全一致项；144 与 1440 不等。该校验不支持单位换算、范围、复杂分数；词语数词未转数字。无数字时 numeric 方法不作拒识，combined 仍依靠阈值，因此 UNKNOWN 并无绝对保证。

其他数值/符号错误（C，全局，已知样本）：

|输入|标准答案|纯 Top1|余弦|
|---|---|---|---:|
|Video encoding format: H.265|VIDEO_CODEC_TYPE_H265|VIDEO_CODEC_TYPE_H264|0.854378|
|Video Codec: Set it to H.265|VIDEO_CODEC_TYPE_H265|VIDEO_CODEC_TYPE_H264|0.858371|
|Exposure Compensation: 1 EV|EXPOSURE_COMPENSATION_TYPE_PLUS_1EV|EXPOSURE_COMPENSATION_TYPE_MINUS_1EV|0.994110|
|Exposure Compensation: 2 EV|EXPOSURE_COMPENSATION_TYPE_PLUS_2EV|EXPOSURE_COMPENSATION_TYPE_MINUS_2EV|0.993580|
|Exposure Compensation: Set it to +2 exposure value|EXPOSURE_COMPENSATION_TYPE_PLUS_2EV|EXPOSURE_COMPENSATION_TYPE_MINUS_2EV|0.895586|
|Sensor sensitivity: I would like 100|ISO_TYPE_100|EXPOSURE_COMPENSATION_TYPE_PLUS_2EV|0.729606|
|Sensor sensitivity: I would like 200|ISO_TYPE_200|EXPOSURE_COMPENSATION_TYPE_PLUS_2EV|0.731044|

曝光 +1/+2 被匹配到负值、H.265 被匹配到 H.264，说明不仅不存在的数值会出错，已支持值的精确符号/数字也可能被相似语义淹没。ISO 的错误同时包含字段误判；只在错误字段内做数值校验会误拒，因此数值 guard 不能替代正确的字段识别。

## 按类别分析（C，全局，纯 Embedding）

|类别|数量|正确率（已知）或未知错误接受率|
|---|---:|---:|
|near_semantic|7|100.00%|
|new_enum|12|100.00%|
|new_member|4|100.00%|
|numeric|80|91.25%|
|standard_synonym|88|82.95%|
|unrelated|3|100.00%|
|unsupported_codec|1|100.00%|
|unsupported_field|3|100.00%|
|unsupported_mode|5|100.00%|
|unsupported_number|21|100.00%|

组合 guard 仍错误接受的未知输入（C，全局）：

|输入|错误接受|余弦|
|---|---|---:|
|White Balance: shade|WHITE_BALANCE_TYPE_FLUORESCENT|0.877139|
|Sharpness: extra high|SHARPNESS_HIGH|0.953173|

## 零样本候选扩展

同一个已加载模型在新增前后的候选集合上重复推理，模型权重未变。新增前排除 1440FPS 与 ImageTextureType，新增后只重算候选向量；这里的“零样本”仅指本地无增训，不能证明预训练语料从未包含这些词或概念。

|场景|阶段|编码|正确数/总数|
|---|---|---|---:|
|new_member|after|A|4/4|
|new_member|after|B|4/4|
|new_member|after|C|4/4|
|new_member|before|A|0/4|
|new_member|before|B|0/4|
|new_member|before|C|0/4|
|new_enum|after|A|12/12|
|new_enum|after|B|12/12|
|new_enum|after|C|12/12|
|new_enum|before|A|0/12|
|new_enum|before|B|0/12|
|new_enum|before|C|0/12|
|naming|schema_0|A|2/3|
|naming|schema_0|B|3/3|
|naming|schema_0|C|3/3|
|naming|schema_1|A|2/3|
|naming|schema_1|B|2/3|
|naming|schema_1|C|2/3|

命名变体实验使用两个分开的候选空间，事先在 data/naming_variants.json 中约定 SHARPNESS_HIGH 与 IMAGE_ACUTANCE_PLUS 均表示强边缘增强。此等价关系仅适用于本合成 fixture，不能推断真实设备也等价；并未根据预测倒推答案。

## 运行成本与时延口径

- 模型目录 128.27 MiB；RAM 总量 16236 MiB。
- 独立虚拟环境目录 4839.2 MiB（包括 CUDA 库）；下载 wheel/分段临时文件不计入，见 downloads 目录。
- 模型加载 943.1 ms（不含 Python/依赖 import，OS 缓存状态未控制）；3 次预热合计 454.4 ms。
- 加载阶段进程 RSS 增量 23.2 MiB；加载后总 RSS 634.6 MiB；预热后 895.7 MiB。
- CUDA tensor allocated 135.4 MiB；reserved 162.0 MiB；峰值 allocated 143.2 MiB。
- RSS 包含 tokenizer/Python 开销；CUDA allocated 不含驱动和 CUDA context，不能相加理解为独占系统显存。
- 每条完整查询与字段分别顺序编码 3 次，先做 3 次预热；CUDA 同步计时。不同策略复用同一查询向量，消除查询编码波动。
- 表中 P50/P95 为各样本重复编码时延中位数的分位数；单条全局时延=完整查询编码+相似度/排序；层级另加字段编码。没有把字段缓存收益或批量吞吐当作单条时延。raw repeat 时间见 CSV。
- 测试使用正在运行桌面程序的共享笔记本 GPU，未锁定频率/功耗、未控制其他进程；时延是本机观测值，不能直接当成生产 SLA。
- 相似度含字段点积、全候选点积、筛选与排序；层级实现仍计算全部候选点积，因此此 P0 不宣称减少了向量计算量。数值 guard/阈值的 Python 决策另记 decision_ms，并纳入端到端时延；不含文件输出和进程启动。

|编码|候选预计算 ms|字段预计算 ms|向量字节数|
|---|---:|---:|---:|
|A|50.123|15.191|87552|
|B|32.775|14.745|87552|
|C|23.673|13.278|87552|

查询时延分解（C 配置，逐样本重复中位数的分位数）：

|检索|组件|P50 ms|P95 ms|
|---|---|---:|---:|
|global|query_encode_ms|11.0617|16.1872|
|global|field_encode_ms|0.0000|0.0000|
|global|similarity_ms|0.0104|0.0183|
|global|decision_ms|0.0085|0.0960|
|hierarchical|query_encode_ms|11.0617|16.1872|
|hierarchical|field_encode_ms|11.0150|15.8931|
|hierarchical|similarity_ms|0.0119|0.0135|
|hierarchical|decision_ms|0.0124|0.0874|

## 典型成功和失败

|输入|标准答案|纯 Top1|分数|结果|
|---|---|---|---:|---|
|Camera Mode: Slow Motion|CAMERA_MODE_TYPE_SLOWMOTION|CAMERA_MODE_TYPE_SLOWMOTION|1.000000|PASS|
|Camera shooting mode: slow-motion video|CAMERA_MODE_TYPE_SLOWMOTION|CAMERA_MODE_TYPE_SLOWMOTION|0.899381|PASS|
|Camera Mode: Set it to slow-motion video|CAMERA_MODE_TYPE_SLOWMOTION|CAMERA_MODE_TYPE_SLOWMOTION|0.942534|PASS|
|Camera shooting mode: I would like Slow Motion|CAMERA_MODE_TYPE_SLOWMOTION|CAMERA_MODE_TYPE_SLOWMOTION|0.951246|PASS|
|Camera shooting mode: continuous photo shooting|CAMERA_MODE_TYPE_BURST|CAMERA_MODE_TYPE_PHOTO|0.830516|FAIL|
|Camera Mode: Set it to continuous photo shooting|CAMERA_MODE_TYPE_BURST|CAMERA_MODE_TYPE_PHOTO|0.851755|FAIL|
|Device operation: convert video encoding|WORKING_MODE_TYPE_TRANSCODE|VIDEO_CODEC_TYPE_H264|0.778174|FAIL|
|Color balance: automatic white balance|WHITE_BALANCE_TYPE_AUTO|WHITE_BALANCE_TYPE_FLUORESCENT|0.841464|FAIL|
|Color balance: overcast lighting|WHITE_BALANCE_TYPE_CLOUDY|WHITE_BALANCE_TYPE_FLUORESCENT|0.801044|FAIL|
|White Balance: Set it to overcast lighting|WHITE_BALANCE_TYPE_CLOUDY|WHITE_BALANCE_TYPE_FLUORESCENT|0.829120|FAIL|
|Color balance: incandescent lighting|WHITE_BALANCE_TYPE_TUNGSTEN|WHITE_BALANCE_TYPE_FLUORESCENT|0.845416|FAIL|
|White Balance: Set it to incandescent lighting|WHITE_BALANCE_TYPE_TUNGSTEN|WHITE_BALANCE_TYPE_FLUORESCENT|0.864521|FAIL|
|Frame Rate: 144 FPS|UNKNOWN|VIDEO_FRAME_RATE_TYPE__1440FPS|0.957961|FAIL|
|Frame Rate: 23 FPS|UNKNOWN|VIDEO_FRAME_RATE_TYPE__24FPS|0.897932|FAIL|
|Frame Rate: 29 FPS|UNKNOWN|VIDEO_FRAME_RATE_TYPE__1440FPS|0.884404|FAIL|
|Frame Rate: 49 FPS|UNKNOWN|VIDEO_FRAME_RATE_TYPE__48FPS|0.871991|FAIL|

## 结论与下一步

**未经增训的 BGE-small-en-v1.5 可以提供英文参数的候选召回，但单独不足以安全支撑完整的自然语言到设备枚举自动映射。**

在本合成留出集，最佳纯 Embedding 配置为 C/global，已知 Top1 88.04%；其未知错误接受率仍为 100.00%，因为强制 Top1 没有表达“不支持”的机制。
C/global 的阈值方法把未知 FAR 降至 42.50%，但已知正确接受率降至 70.11%。组合 guard 的 FAR 为 5.00%，已知正确接受率 71.74%，已知误拒 28.26%，仍不适合无兜底自动执行。
层级 C 的字段准确率为 95.65%，低于全局的 96.74%；最终 Top1 为 85.87%，因此本实验未证明两级匹配更好。
精确数值与支持性属于离散约束，余弦距离不能证明相等；词义近似、字段混淆与不规范复合命名又会引入排名错误。A/B/C 的差别说明候选表述质量有影响，但 C 含人工先验；层级字段预测也会传播错误。

建议下一阶段先获取真实设备 Proto、成员含义说明和独立人工标注查询，复验字段混淆及负例覆盖。将 Embedding 定位为候选召回，保留已支持集合、物理数值/单位验证与 UNKNOWN 返回；非数值未知需补充代表性负例再标定。本轮没有比较更大模型，不能断言换大模型会解决数值相等或未知检测问题，也不据此直接推荐。

## 原始证据

- results.csv：全部 dev/test × 3 编码 × 2 检索 × 4 策略，共 8064 行；保留原始 Top3 与最终决策。
- results/summary.csv、summary.json：独立测试集 24 个配置的指标。
- results/thresholds.json：6 个仅由开发集选择的阈值。
- results/zero_shot.csv：插入候选前后与命名变体实测。
- results/environment.json、precompute.json：环境、哈希、成本与预计算。
- data/dataset.json、camera.proto：固定标签与协议样例。

## 资料

- [模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5) （短对称匹配不必加检索 instruction，余弦不能当概率）。
- [PyTorch CUDA 12.1 2.5.1 分发](https://pytorch.org/get-started/previous-versions/)。
- 本报告全部实验数值来自本地执行，公开模型卡不是评测结果来源。
