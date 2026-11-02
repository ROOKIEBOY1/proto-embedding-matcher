# proto-embedding-matcher

使用 **BGE-small-en-v1.5 + Sentence Transformers**，将英文自然语言相机参数匹配到 Proto 枚举，并评测纯 Embedding 的语义匹配、精确数值、未知拒识和零样本扩展能力。

```text
Camera Mode: Slow Motion → CameraModeType / CAMERA_MODE_TYPE_SLOWMOTION = 0
Frame Rate: 48 FPS       → VideoFrameRateType / VIDEO_FRAME_RATE_TYPE__48FPS = 4
Frame Rate: 144 FPS      → 示例协议不支持，启用数值校验后返回 UNKNOWN
```

这是一个**本地 P0 评测项目**，核心集中在单个 Python 文件中。模型不增训，推理不调用在线 Embedding API。

> 当前协议和数据是合成样例，不代表真实产品的完整协议。实测显示模型适合候选召回，但不足以单独保证精确参数映射和未知拒识。余弦相似度不是正确概率。

## 1. 功能与文件结构

- 解析简单 Proto enum，保留类型、成员名称和 wire ID。
- 对比 A 原始名称、B 自动拆词、C 英文标准化三种候选编码。
- 对比全局匹配与先匹配字段类型、再匹配枚举成员的两级匹配。
- 输出真实 Top-3 余弦、编号、margin 和最终决定。
- 对比强制 Top-1、开发集阈值、数值精确校验与组合 guard。
- 测试新增成员、新增 enum 和不同命名方式，无需增训。
- 自动选择可用 CUDA，支持强制 CPU；记录预热、重复时延和内存。
- 导出原始 CSV、汇总、字段混淆、分类结果和 Markdown 报告。
- 审计 CSV，并重新调用真实模型复算部分排名与分数。

```text
proto-embedding-matcher/
├── proto_embedding_p0.py       # 核心：解析、描述、模型、排名、校验、评测、CLI、报告
├── build_fixtures.py           # 生成合成协议与预先标注的数据
├── test_proto_embedding_p0.py  # 无需模型下载的逻辑与数据一致性测试
├── verify_results.py           # 结果审计和真实模型分数重放
├── download_model.ps1          # 固定版本模型下载与权重 SHA256 校验
├── download_torch.py           # 可选：Windows CUDA wheel 分段下载修复工具
├── requirements.txt            # 普通依赖；PyTorch 按 CPU/GPU 单独安装
├── requirements-lock.txt       # 原始 Windows CUDA 实验的完整版本
├── data/
│   ├── camera.proto            # 11 个 enum、46 个成员，Proto2 合成样例
│   ├── dataset.json            # 固定答案、enum 类型、编号、来源和类别
│   ├── naming_variants.json    # 不同命名实验的语义约定
│   └── ANNOTATION_NOTES.md      # 标注约定和关键样本核对说明
├── benchmarks/                 # 已执行实验的汇总、阈值和零样本结果快照
├── REPORT.md                   # 实验详细报告，重跑会更新
├── models/                     # 下载后生成，不提交 Git
├── results/                    # 运行后生成，不提交 Git
└── results.csv                 # 全量逐样本结果，运行后生成，不提交 Git
```

仓库不包含权重、虚拟环境、安装包、缓存、本机日志和约 6MB 的全量原始 CSV。这些可以在本地重建。

## 2. 安装环境（Windows PowerShell）

推荐 Python **3.12**，本次实测为 3.12.14、Windows 11。以下命令需要 Git 和 Python 3.12；未验证 Linux/macOS 的完整安装流程。

### 2.1 克隆并创建虚拟环境

```powershell
git clone https://github.com/ROOKIEBOY1/proto-embedding-matcher.git
cd proto-embedding-matcher
py -3.12 -m venv .venv
```

所有命令直接使用虚拟环境解释器，无需激活环境或修改系统执行策略。
若 `py -3.12` 不可用，可用自己的 Python 3.12 可执行文件完整路径创建 `.venv`。

### 2.2 选择 CPU 或 NVIDIA GPU（二选一）

CPU 安装，不需要显卡和 CUDA：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU 安装，与原实验相同的 CUDA 12.1 构建：

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

原实验使用 RTX 3060 Laptop 6GB、驱动 546.30。显卡驱动需兼容所选 CUDA wheel，不必单独安装完整 CUDA Toolkit。
已安装 CUDA wheel 的环境也可通过 `--device cpu` 强制在 CPU 上推理。

### 2.3 安装普通依赖并检查

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

关键版本：Sentence Transformers 3.4.1、Transformers 4.48.3、NumPy 2.2.2。
`requirements.txt` 不固定 PyTorch 的 CPU/CUDA 构建，使用上一步安装的 torch。
`requirements-lock.txt` 记录原实验全部版本，包含 `torch==2.5.1+cu121`；CPU 用户不要直接套用 CUDA 锁定文件。

## 3. 下载并验证模型

模型：[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)。固定 revision：

```text
5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
```

从官方端点下载：

```powershell
& .\download_model.ps1 -Endpoint "https://huggingface.co"
```

如果 PowerShell 阻止脚本执行，可仅为这一次子进程使用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\download_model.ps1 -Endpoint "https://huggingface.co"
```

官方直连失败时，使用脚本默认镜像：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\download_model.ps1
```

脚本保存到 `models/bge-small-en-v1.5/`，包括 safetensors、tokenizer、池化配置和 `REVISION`，并校验权重 SHA256：

```text
3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad
```

模型目录约 128.27 MiB。推理强制 `local_files_only=True`、`use_safetensors=True` 和离线环境变量，模型准备好后可断网使用。本地元数据缓存写入项目 `.cache/huggingface`。

## 4. 快速开始

### 4.1 检查协议与真实模型加载

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --inspect
```

输出 enum/成员数量和 A/B/C 候选描述，再实际加载模型、编码一条文本。
最后应看到实际设备、384 维向量和接近 1 的归一化范数。

### 4.2 测试一条输入

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Camera Mode" --value "Slow Motion"
```

默认 **C + 全局匹配 + 强制 Top-1**。输出 Top-3 名称、编号、真实余弦、第一二名差距和最终决定。
交互输入没有预设答案，因此不显示 PASS/FAIL。

### 4.3 使用 CPU

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Frame Rate" --value "48 FPS" --device cpu
```

默认 `--device auto` 优先选择可用 CUDA，否则 CPU。显式指定 `cuda` 但不可用时不会静默回退。

### 4.4 执行完整评测，生成开发集阈值

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --eval --csv results.csv
```

终端默认展示 C/global/forced 的每条测试输入，CSV 保留全部配置和开发/测试样本。
仅看进度及最终汇总可加 `--quiet`：

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --eval --csv results.csv --quiet
```

**完整评测会更新 REPORT.md、results 目录的汇总以及指定 CSV。** 保留旧报告请先备份或查看 Git 历史。
`--csv` 只改变逐样本 CSV 路径，不会隔离报告、阈值或汇总目录。

### 4.5 启用拒识校验

先运行上一步生成 `results/thresholds.json`，再测试：

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Frame Rate" --value "144 FPS" --guard
```

在内置协议中，原始 Top-1 是 1440FPS，但没有精确的 144FPS 成员，最终应为 UNKNOWN。
`--guard` 表示开发集阈值加全 enum 数值校验，并没有调用其他模型。

### 4.6 两级匹配及三种编码对比

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Frame Rate" --value "48 FPS" --hierarchical --guard
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Camera Mode" --value "standard video" --strategy A
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Camera Mode" --value "standard video" --strategy B
.\.venv\Scripts\python.exe proto_embedding_p0.py --field "Camera Mode" --value "standard video" --strategy C
.\.venv\Scripts\python.exe proto_embedding_p0.py --eval --hierarchical --guard --quiet
```

## 5. A/B/C 和检索方式

三种策略使用相同模型、候选集合和查询，只改变候选文本及类型文本：

|策略|候选示例|处理|
|---|---|---|
|A|`CAMERA_MODE_TYPE_SLOWMOTION`|直接编码原始名称|
|B|`Camera Mode Type: Slowmotion`|去公共前缀，拆下划线、CamelCase、数字到单位边界|
|C|`Camera Mode: Slow Motion`|在 B 上去 Type，应用公开的小型术语规则|

C 包括 slowmotion→Slow Motion、videonormal→Normal Video、timelapse→Time Lapse，保留 daylight/MJPEG 的规范形式；MINUS_/PLUS_/ZERO_ 转为带符号数值。完整规则见 `COMPOUNDS` 和 `value_text()`。C 包含人工先验，不能将全部收益算作模型泛化。

- 全局：完整 `field: value` 与全部候选比较。
- 两级：先只编码字段，选 enum，再用完整查询在此 enum 内排名。
- 模型输出 384 维、L2 归一化 float32 向量，点积即余弦；短对称参数匹配不加检索 instruction。
- wire ID 不是物理值，例如 48FPS 的编号是 4，不是 48。

## 6. 拒识和阈值

|方法|逻辑|限制|
|---|---|---|
|forced|无条件选纯排名第一名|未知输入也会被接受|
|threshold|低于开发集阈值则 UNKNOWN|高分也可能数字错误|
|numeric|检查预测 enum 的全部候选是否数值一致|无数字时不提供拒识能力|
|combined|数值校验 + 阈值，CLI 对应 --guard|仍可能误拒已知、误接受非数值未知|

数值校验不只看 Top-3；精确数值成员即使排在前三之外，也能被选中。没有精确项则 UNKNOWN。
支持十进制和符号，48 与 48.0 相等；不处理单位换算、英文数词、范围或工作模式组合约束。
校验只在模型预测的 enum 内进行，字段选错仍可能导致错误或误拒。

正确身份需要 **enum 类型 + 成员名称** 同时一致；不同类型即使编号相同也算不同答案。
交互模式的 accepted 仅表示策略接受，不是证明真实设备必然支持。

阈值对每种编码/检索配置独立标定，只使用开发集，最大化“已知接受率和未知拒绝率的平均值”，同分取较高阈值。
测试集不参与选择。更换模型、候选或数据后需重新标定。内置标签和阈值不能直接用于其他协议。

## 7. 数据、完整流程与评测矩阵

46 个成员分别生成 2 条开发表达、4 条测试表达：

|划分|已支持|未知|合计|
|---|---:|---:|---:|
|开发集|92|20|112|
|测试集|184|40|224|

每条样本带 id、split、field、value、expected_enum、expected、expected_number、category、source、template。
正例标明模板生成，负例是人工编写的合成样例；没有真实业务专家独立验收的声明。
开发和测试查询不重叠，正例共享候选标签但表达模板不同，未知数值分开设置。

```text
固定标准答案 → 解析 Proto → A/B/C 描述 → 预计算候选向量
→ 开发集选阈值 → 独立测试 → 新增候选零样本测试 → CSV/报告 → 分数重放审计
```

新增成员/类型只改变候选集合，模型权重不变。命名等价关系在预测前写入 naming_variants.json，不根据预测倒推答案。
重新生成 fixture 会覆盖 data 下三个生成文件：

```powershell
.\.venv\Scripts\python.exe build_fixtures.py
```

完整矩阵为 **3 编码 × 2 检索 × 4 决策 = 24 配置**，336 条开发/测试样本共生成 **8064 行**。
`--strategy`、`--hierarchical` 和 `--guard` 在评测模式只控制逐条展示，不缩小评测矩阵。

## 8. 输出、指标与验证

|输出|内容|
|---|---|
|results.csv|答案、预测类型/名称/编号、原始 Top-3、余弦、margin、拒识、正确性和重复时延|
|results/summary.csv、summary.json|仅独立测试集的 24 个配置指标|
|results/thresholds.json|6 个开发集阈值|
|results/zero_shot.csv|新增候选前后及不同命名结果|
|results/by_category.csv|按类别指标|
|results/field_confusion.csv|字段混淆计数|
|results/environment.json|实际设备、版本、模型/数据哈希和内存，包含本机路径|
|results/precompute.json|候选与字段预计算耗时和向量大小|
|results/verification.json|审计结果和真实模型分数重放|
|REPORT.md|完整中文分析|

指标口径：

- **Top-1 Accuracy**：已支持样本最终正确接受率，误拒也计错；raw_top1_accuracy 单独保留纯排名指标。
- **Top-3 Recall**：已支持样本正确候选进入纯排名前三的比例，与 guard 决策分开。
- **Field Accuracy**：已支持样本类型正确率；全局取 Top-1 所属 enum。
- **False Acceptance Rate**：未知样本被错误接受的比例。
- **Unknown Rejection Recall**：未知被拒比例，等于 1−FAR。
- **Known Rejection Rate**：已支持样本被误拒比例。
- **P50/P95**：每条重复编码中位数的分位数，先预热 3 次，默认每条重复 3 次。

模型加载、候选预计算、完整查询编码、字段编码、相似度排序、决策分别记录。
GPU 同步计时，不含进程启动和文件输出；层级另加字段编码，当前仍计算全部候选点积。
桌面共享 GPU 的时延不是生产 SLA。进程 RSS、加载增量和 CUDA allocated/reserved 分开记录，CUDA 张量内存不包含所有驱动开销。

逻辑与数据测试，无需下载模型：

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

完整评测后，检查 CSV 和真实分数：

```powershell
.\.venv\Scripts\python.exe verify_results.py
```

审计检查 8064 行唯一性、排序、分数范围、身份判定、6 个阈值重算，并用模型重放 48FPS/144FPS 的 A/B/C 和全局/层级 Top-3。
审计脚本读取默认根目录 results.csv；若先前用了其他名字，请复制为 results.csv 后再审计。

## 9. 已执行基线

环境：Windows 11、Python 3.12.14、PyTorch 2.5.1+cu121、RTX 3060 Laptop 6GB。完整评测两次，准确率和拒识率一致。

|编码|全局 Top-1|全局 Top-3|两级 Top-1|
|---|---:|---:|---:|
|A|85.33%|96.74%|82.61%|
|B|86.41%|95.65%|84.24%|
|C|88.04%|95.65%|85.87%|

C/global 决策对比：

|方法|已知正确接受率|未知误接受率|已知误拒率|
|---|---:|---:|---:|
|纯模型|88.04%|100.00%|0.00%|
|阈值|70.11%|42.50%|28.26%|
|数值校验|90.76%|37.50%|1.09%|
|组合 guard|71.74%|5.00%|28.26%|

真实失败：144FPS→1440FPS，C 余弦 **0.957961**；+1EV→−1EV，余弦 **0.994110**。
高分不能证明数字/符号相等。组合校验减少误接受，但误拒较高，非数值未知仍有漏过。
新增 1440FPS 的 4 条表达、新增 ImageTexture 的 12 条表达在添加候选后全部匹配，无增训；不同命名仍有失败。

GPU C/global P50/P95 约 **11.08/16.20 ms**，模型约 **128.27 MiB**，CUDA allocated 约 **135.39 MiB**。
这是 GPU 时延，不是 CPU 时延。完整 CUDA 虚拟环境约 4.73 GiB，远大于模型本身。

详细案例见 [REPORT.md](REPORT.md)、[基线汇总](benchmarks/summary.csv)。benchmarks 是上传时快照，重跑不会覆盖；完整原始 CSV 需在本地重建。
本模型适合候选召回实验，尚不适合无兜底自动执行。没有比较更大模型，不声称换大模型能解决精确参数问题。

## 10. 使用自己的 Proto

```powershell
.\.venv\Scripts\python.exe proto_embedding_p0.py --proto ".\my_camera.proto" --field "Frame Rate" --value "48 FPS"
```

当前解析器只覆盖简单 enum、注释、有符号 ID 和成员 options；不解析 import、package/嵌套作用域，不是 protoc 的替代品。
同名 enum 会报错，避免静默混淆。真实协议还需审核保留值、占位成员、别名与设备能力。
内置评测标签和阈值只对应 data/camera.proto，自定义协议不能直接用 --eval 或 --guard。

完整真实协议的下一步应是完善解析、确认字段与设备能力、独立标注数据并重新标定。
**Proto 定义了某成员，不代表每个设备型号、固件和工作模式都支持它。**

## 11. CLI 参数

|参数|含义|
|---|---|
|--inspect|候选描述和真实离线加载验证|
|--eval|完整 24 配置评测|
|--csv PATH|逐样本 CSV 路径|
|--field TEXT / --value TEXT|交互字段和值，必须同时提供|
|--strategy A/B/C|交互候选编码；评测时控制展示，默认 C|
|--hierarchical|交互启用两级匹配；评测时控制展示|
|--guard|组合拒识，需先标定|
|--device auto/cpu/cuda|实际设备|
|--model PATH|本地模型路径|
|--proto PATH|简单 Proto 路径|
|--repeats N|每条查询重复编码次数，默认 3，至少 1|
|--quiet|评测不逐条展示|

## 12. 常见问题

**模型离线加载失败**：先下载模型，确认 model.safetensors、modules.json、1_Pooling/config.json、tokenizer 和 REVISION 齐全。
--model 可以指定其他目录，但完整评测环境记录仍从默认模型目录读取部分版本/哈希，复现时请保持默认目录。

**guard 找不到阈值**：先执行 --eval。benchmarks/thresholds.json 是历史证据，不自动作为本地阈值。

**CUDA 不可用**：用 --device cpu，并检查 torch 构建及显卡驱动，无需修改系统 Python。

**包源下载慢**：普通依赖可添加 --index-url https://pypi.tuna.tsinghua.edu.cn/simple；PyTorch 按 CPU/CUDA 专用索引安装。

**CUDA wheel 整包下载失败**：可选运行 download_torch.py，仅适用于 Windows/Python 3.12/torch 2.5.1+cu121；从交大镜像分段下载并验证官方 SHA256，再安装 downloads 下生成的 wheel。正常下载无需此工具。

**CPU 评测是否覆盖 GPU 报告**：会更新默认报告、阈值和汇总，先备份。--csv results_cpu.csv 只隔离逐样本 CSV。

## 13. 参考

- [BGE 模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [PyTorch CPU/CUDA 安装版本](https://pytorch.org/get-started/previous-versions/)
- [详细实验报告](REPORT.md)

模型及其权重遵循发布方许可，本仓库不重新分发模型权重。
