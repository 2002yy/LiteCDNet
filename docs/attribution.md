# 代码来源说明

本仓库是作者在本科毕设阶段整理出的公开版实验仓库，包含原创整理代码，也包含为复现实验而集成的对比模型实现。

## 1. 作者整理与公开版新增内容

以下内容主要是为本项目训练、评估、公开发布而整理或新增：

- `README.md`
- `requirements.txt`
- `docs/README.md`
- `docs/method.md`
- `docs/reproducibility.md`
- `src/data_config.py`
- `src/utils_.py`
- `src/main_LiteCDNET.py`
- `src/main_ablation.py`
- `src/ablation/`
- 公开版目录整理与 `src/`、`assets/` 结构

这些部分主要负责：

- 公开仓库结构收敛
- 数据路径配置
- LiteCDNet 主训练入口
- A0-A7 消融实验入口与辅助流程
- 面向外部读者的复现说明

## 2. 对比模型与适配实现

`src/compare/` 中保留了多个对比模型实现，用于在统一训练框架下做基线比较。这些文件对应的模型思想来源于各自论文或公开复现实现，仓库中主要做了以下类型的适配：

- 接入当前项目的 `define_G` 与训练入口
- 对输入输出接口进行统一
- 补齐或调整预训练权重加载方式
- 为当前数据集与实验脚本做工程化改造

涉及的目录与文件包括但不限于：

- `src/compare/A2Net.py`
- `src/compare/A2Net_v2.py`
- `src/compare/ChangeFormer.py`
- `src/compare/DMINet.py`
- `src/compare/DTCDSCN.py`
- `src/compare/FC_EF.py`
- `src/compare/FC_Siam_conc.py`
- `src/compare/FC_Siam_diff.py`
- `src/compare/IFNet.py`
- `src/compare/NestedUNet.py`
- `src/compare/SNUNet.py`
- `src/compare/TFI_GR.py`

这些代码应理解为“项目内对比实验所用实现”，不应被简单表述为全部为作者从零原创。

## 3. LiteCDNet 与 SEIFNet 相关实现

以下部分与本项目核心研究实现关系最密切：

- `src/compare/LiteCDNET.py`
- `src/models/networks.py`
- `src/models/trainer.py`
- `src/models/evaluator.py`
- `src/models/losses.py`

其中一部分代码在原始实验工作区中逐步演化而来，后续又为了公开发布进行了目录重组、路径修正和工程化清理。

## 4. 使用与引用建议

如果你在阅读、复现或二次整理本仓库时需要说明来源，建议使用如下表述：

- LiteCDNet 公开版仓库由作者基于本科毕设实验代码整理而成
- 对比模型部分为论文/公开实现基础上的项目内适配版本
- 公开版新增了仓库结构整理、复现文档与数据路径配置

## 5. 说明边界

本说明的目标是帮助读者区分：

- 哪些部分主要是本项目公开整理代码
- 哪些部分属于对比模型集成与适配

如果未来继续公开更细粒度的论文引用信息，建议在对应模型文件附近补充更精确的论文标题、作者和仓库链接。
