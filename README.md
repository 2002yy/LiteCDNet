# LiteCDNet

公开版 LiteCDNet 仓库，聚焦于遥感影像变化检测的训练、评估与消融复现。

这个版本从原始毕设工作区中收敛而来，只保留对外复现需要的内容：

- `src/`：训练、评估、模型定义与消融代码
- `docs/`：方法说明与复现实验说明
- `assets/`：README 和文档中用到的少量示意图

数据集、模型权重、可视化输出、论文排版脚本、答辩材料和个人工作目录均不包含在公开仓库中。

## 项目概览

LiteCDNet 面向双时相遥感图像变化检测，公开版仓库保留了以下能力：

- LiteCDNet 主模型训练与测试
- SEIFNet、A2Net、IFNet、ChangeFormer 等对比模型训练入口
- LiteCDNet A0-A7 消融实验入口
- 统一的数据加载、评估与日志流程

![LiteCDNet Architecture](assets/litecdnet-architecture.png)

## 环境准备

建议使用 Python 3.10 或 3.11，并提前安装与你设备匹配的 PyTorch。

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你需要 CUDA 版本的 PyTorch，请先按照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的说明安装对应版本，再执行 `pip install -r requirements.txt`。

## 数据准备

仓库默认使用相对路径约定，不再依赖作者本机磁盘路径。

默认目录结构如下：

```text
data/
  LEVIR/
  DSIFN_256/
  SYSU-CD/
  LEVIR-CD+_256/
  Big_Building_ChangeDetection/
  GZ/
  WHU-CUT/
```

你可以任选以下方式指定数据路径：

1. 按默认结构把数据放到仓库根目录下的 `data/`
2. 为脚本传入 `--data_root <你的数据路径>`
3. 设置环境变量

通用环境变量：

```powershell
$env:LITECDNET_DATA_ROOT="D:\\datasets\\LEVIR"
```

数据集专属环境变量示例：

```powershell
$env:LITECDNET_LEVIR_ROOT="D:\\datasets\\LEVIR"
```

## 常用命令

LiteCDNet 主训练：

```bash
python -X utf8 src/main_LiteCDNET.py --data_name LEVIR
```

对比模型训练：

```bash
python -X utf8 src/main_train.py --net_G SEIFNet --data_name LEVIR
python -X utf8 src/main_train.py --net_G ChangeFormer --data_name LEVIR
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

如果数据不在默认目录下，可以追加：

```bash
--data_root D:\\datasets\\LEVIR
```

## 结果与说明

- 本仓库不包含训练好的 checkpoint
- 训练输出默认写入 `checkpoints/`、`checkpoints_ablation/`、`vis/` 和 `vis_ablation/`
- 这些目录默认被 `.gitignore` 忽略，避免公开仓库膨胀

README 中附带了少量示意图，完整论文图表与答辩材料不在公开版中分发。

![Qualitative Results](assets/qualitative-results.png)

## 额外说明

- `DMINet` 相关代码可能需要额外的本地预训练权重，例如 `src/pretrain_model/resnet18-5c106cde.pth`
- 公开版重点维护的入口是 `src/main_LiteCDNET.py`、`src/main_train.py`、`src/main_ablation.py` 和 `src/eval_cd.py`
- 更细的复现说明见 [docs/reproducibility.md](docs/reproducibility.md)

## 文档

- [文档索引](docs/README.md)
- [代码来源说明](docs/attribution.md)
- [方法说明](docs/method.md)
- [复现实验说明](docs/reproducibility.md)

## 代码来源说明

本仓库中的代码由以下几类内容组成：

- LiteCDNet 相关主流程、公开版仓库整理、数据路径配置与消融组织代码
- 为对比实验保留的基线模型实现
- 在公开实现或论文复现思路基础上整理、适配到本项目训练框架中的模块

为避免混淆，来源和适配范围已在 [docs/attribution.md](docs/attribution.md) 中单独说明。
