# LiteCDNet

LiteCDNet 是一个面向遥感变化检测实验的公开整理版仓库，聚焦训练、评估、消融和基线对比复现。当前公开版从原始毕设工作区中收敛而来，只保留对外复现真正需要的代码、说明文档和少量示意图。

公开版包含：
- `src/`：训练、评估、模型定义、消融与辅助脚本
- `docs/`：方法说明、复现实验说明、来源说明、细粒度引用清单
- `assets/`：README 与文档中使用的少量示意图

公开版不包含：
- 数据集
- 训练得到的 checkpoint
- 大量可视化输出与中间实验产物
- 答辩材料、论文写作草稿、个人工作目录镜像

## 项目概览

本仓库当前主要保留以下能力：
- LiteCDNet 主模型训练与评估
- SEIFNet 与多种对比模型的统一训练入口
- LiteCDNet A0-A7 消融实验入口
- 统一的数据路径配置、日志组织与评估流程

![LiteCDNet Architecture](assets/litecdnet-architecture.png)

## Quick Start

如果你是第一次进入这个仓库，建议按下面顺序开始：

1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据：按下文的数据目录结构放置 LEVIR-CD 或其他支持数据集
3. 训练主模型：`python -X utf8 src/main_LiteCDNET.py --data_name LEVIR`
4. 评估主模型：`python -X utf8 src/eval_cd.py --project_name LEVIR_LiteCDNet_BCEDiceBoundary0.3_AdamW_Cosine_150 --data_name LEVIR`
5. 查看方法与复现说明：`docs/method.md` 与 `docs/reproducibility.md`

对于只想快速了解仓库结构的读者，可以先阅读本页的主结果、复杂度统计和可视化部分；对于准备复现训练的读者，建议进一步阅读 `docs/reproducibility.md`。

## Code Map

下面列出最核心的公开入口，方便第一次进入仓库时快速定位：

| Path | Role |
| --- | --- |
| `src/main_LiteCDNET.py` | LiteCDNet 主模型训练入口 |
| `src/eval_cd.py` | LiteCDNet 主模型评估入口 |
| `src/main_train.py` | SEIFNet、A2Net、ChangeFormer 等对比模型统一训练入口 |
| `src/main_ablation.py` | A0-A7 消融实验统一入口 |
| `src/compare/LiteCDNET.py` | LiteCDNet 主模型实现 |
| `src/models/networks.py` | 模型调度与 `SEIFNet` 等核心网络定义 |
| `src/models/trainer.py` | 训练流程封装 |
| `src/models/evaluator.py` | 评估流程封装 |
| `src/datasets/CD_dataset.py` | 变化检测数据集读取逻辑 |
| `src/data_config.py` | 数据路径解析与数据集目录映射 |
| `src/utils/metrics.py` | 常用指标计算工具 |
| `src/ablation/` | 消融变体、复杂度统计与运行器 |

## 仓库结构

```text
LiteCDNet/
├─ assets/                     # README / docs 使用的少量图示
├─ docs/                       # 面向公开读者的说明文档
├─ src/
│  ├─ ablation/               # LiteCDNet 消融变体与运行逻辑
│  ├─ compare/                # 对比模型与相关辅助实现
│  ├─ datasets/               # 数据集加载逻辑
│  ├─ misc/                   # 通用辅助模块
│  ├─ models/                 # 训练器、评估器、模型调度
│  ├─ scripts/                # 报表/统计/快捷脚本
│  └─ utils/                  # 工具函数
├─ README.md
├─ NOTICE.md
├─ requirements.txt
└─ CITATION.cff
```

## 环境准备

建议使用 Python 3.10 或 3.11，并提前安装与你设备匹配的 PyTorch。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你需要 CUDA 版本的 PyTorch，请先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装对应版本，再执行 `pip install -r requirements.txt`。

## 数据准备

公开版已经移除作者本机硬编码路径。默认约定是把数据放在仓库根目录下的 `data/` 目录：

```text
data/
├─ LEVIR/
├─ DSIFN_256/
├─ SYSU-CD/
├─ LEVIR-CD+_256/
├─ Big_Building_ChangeDetection/
├─ GZ/
└─ WHU-CUT/
```

你可以通过以下任一方式指定数据路径：

1. 直接按默认结构放到仓库根目录下的 `data/`
2. 给训练/评估脚本传入 `--data_root <你的数据目录>`
3. 设置环境变量 `LITECDNET_DATA_ROOT`
4. 针对单个数据集设置更细粒度环境变量，例如 `LITECDNET_LEVIR_ROOT`

PowerShell 示例：

```powershell
$env:LITECDNET_DATA_ROOT="D:\datasets"
$env:LITECDNET_LEVIR_ROOT="D:\datasets\LEVIR"
```

## 常用命令

LiteCDNet 主模型训练：

```bash
python -X utf8 src/main_LiteCDNET.py --data_name LEVIR
```

统一训练入口中的对比模型示例：

```bash
python -X utf8 src/main_train.py --net_G SEIFNet --data_name LEVIR
python -X utf8 src/main_train.py --net_G ChangeFormer --data_name LEVIR
python -X utf8 src/main_train.py --net_G A2Net --data_name LEVIR
```

LiteCDNet 消融实验：

```bash
python -X utf8 src/main_ablation.py --ablation_case full --data_name LEVIR
python -X utf8 src/main_ablation.py --ablation_case no_context --data_name LEVIR
```

模型评估：

```bash
python -X utf8 src/eval_cd.py --project_name LEVIR_LiteCDNet_BCEDiceBoundary0.3_AdamW_Cosine_150 --data_name LEVIR
```

如果数据不在默认目录下，可以额外传入：

```bash
--data_root D:\datasets
```

## LEVIR-CD 主结果

下表整理了论文主结果口径下的 LEVIR-CD 定量结果，用于展示 LiteCDNet 与代表性强基线 SEIFNet 的精度-复杂度折中。

| Model | FLOPs (G) | Params (M) | Acc | mIoU | mF1 | IoU(change) | F1(change) | Precision(change) | Recall(change) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEIFNet | 8.37 | 27.91 | 0.97821 | 0.80942 | 0.88507 | 0.64151 | 0.78161 | 0.79851 | 0.76541 |
| LiteCDNet | 2.14 | 2.47 | 0.97837 | 0.81035 | 0.88575 | 0.64321 | 0.78287 | 0.80127 | 0.76530 |

说明：
- LiteCDNet 的 FLOPs 和 Params 来自当前项目实现下的本地前向统计。
- SEIFNet 的 FLOPs 和 Params 用于公开结果对照展示，不应解读为完全同条件复现实验下的严格速度结论。

主结果对照图：

![LEVIR-CD Parameter-IoU Comparison](assets/levir-parameter-iou-comparison.png)

## 复杂度统计与消融

标准 A0-A7 消融实验主要用于分析 LiteCDNet 各关键模块和训练策略的贡献。这里保留一份适合 README 快速浏览的摘要表。

| Code | Setting | Params (M) | FLOPs (G) | Best val mIoU | Best val mF1 | Delta mIoU vs A0 |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | Full LiteCDNet | 2.47 | 2.14 | 0.81628 | 0.88959 | — |
| A1 | `abs_diff` replaces DiffFusion | 2.24 | 2.07 | 0.76093 | 0.84729 | -0.05535 |
| A2 | Remove LiteContext | 2.23 | 2.05 | 0.81338 | 0.88749 | -0.00290 |
| A3 | Concat decoder | 2.73 | 2.68 | 0.82246 | 0.89403 | +0.00618 |
| A4 | Remove boundary loss | 2.47 | 2.14 | 0.79182 | 0.87144 | -0.02446 |
| A5 | Remove deep supervision | 2.47 | 2.14 | 0.80894 | 0.88425 | -0.00734 |
| A6 | `boundary=0.5` | 2.47 | 2.14 | 0.80940 | 0.88459 | -0.00688 |
| A7 | Adjust multi-scale loss weights | 2.47 | 2.14 | 0.81207 | 0.88656 | -0.00421 |

说明：
- A0-A7 采用的是统一验证集口径，用于模块贡献分析，不等同于上面的 LEVIR-CD 主测试结果表。
- 从复杂度角度看，A1 和 A2 最轻；A3 的参数量和 FLOPs 明显更高，说明拼接式解码更重。

复杂度总览图：

![LiteCDNet Complexity Overview](assets/complexity-overview.png)

## 预测可视化

下面给出一张公开版单样本预测示意图，方便快速了解模型输出形态。当前仓库不打包大规模可视化结果集，只保留少量 README 展示图。

![LiteCDNet Prediction Example on LEVIR-CD](assets/prediction-example-levir.jpg)

## 结果与公开边界

- 本仓库不分发训练好的 checkpoint
- 训练输出默认会写入 `checkpoints/`、`checkpoints_ablation/`、`vis/` 和 `vis_ablation/`
- 这些目录默认被 `.gitignore` 忽略，避免公开仓库膨胀
- README 仅保留少量展示图，完整论文图表与答辩材料不在公开版中提供

## 文档导航

- [文档索引](docs/README.md)
- [中英双语项目简介](docs/project-overview-bilingual.md)
- [方法说明](docs/method.md)
- [复现实验说明](docs/reproducibility.md)
- [来源说明](docs/attribution.md)
- [细粒度论文/仓库引用清单](docs/references.md)
- [LICENSE 选择建议](docs/license-options.md)
- [首个公开版 Release Notes](docs/release-notes-v1.0.0.md)
- [第三方代码说明](NOTICE.md)

## 公开论文

仓库中提供了一份适合公开分发的论文副本，已去除封面与摘要页中的个人信息：

- [公开版论文 DOCX](publications/LiteCDNet_thesis_public_redacted.docx)
- [公开版论文 PDF](publications/LiteCDNet_thesis_public_redacted.pdf)

## 代码来源与引用

本仓库中的代码由三部分组成：

1. LiteCDNet / SEIFNet / 消融实验相关的项目内实现与公开整理代码
2. 为基线比较保留的对比模型实现
3. 为统一训练框架而做的接口适配、路径整理和工程化清理

阅读或引用时建议区分两层来源：
- 仓库级说明：见 [docs/attribution.md](docs/attribution.md)
- 文件级论文/仓库映射：见 [docs/references.md](docs/references.md)

如果你使用了本仓库中的具体对比模型，请同时引用对应模型论文或官方仓库；如果你引用的是整个公开整理版仓库，请参考 [CITATION.cff](CITATION.cff)。

## License and Release

当前公开仓库尚未附带最终 `LICENSE` 文件。与许可选择相关的背景说明和候选方案整理在 [docs/license-options.md](docs/license-options.md) 中；首个公开版发布说明见 [docs/release-notes-v1.0.0.md](docs/release-notes-v1.0.0.md)。

## Notes

- `DMINet` 等部分对比模型可能需要额外的本地预训练权重，例如 `src/pretrain_model/resnet18-5c106cde.pth`
- 公开仓库当前重点维护的主入口为 `src/main_LiteCDNET.py`、`src/main_train.py`、`src/main_ablation.py` 和 `src/eval_cd.py`
- 更细的实验边界、输入组织、日志约定和复现说明见 [docs/reproducibility.md](docs/reproducibility.md)
