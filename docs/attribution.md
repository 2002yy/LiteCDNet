# 代码来源说明

本仓库是作者在本科毕设阶段整理出的公开实验仓库，既包含项目内原创整理代码，也包含为对比实验而统一接入的公开模型实现。这个文档的目标不是逐行追溯每段代码，而是帮助读者理解“哪些部分主要属于本项目整理内容，哪些部分属于公开方法的项目内适配版本”。

## 1. 项目内整理与公开版新增内容

以下内容主要服务于本仓库的公开发布、统一训练流程和复现实验说明：

- `README.md`
- `NOTICE.md`
- `CITATION.cff`
- `requirements.txt`
- `docs/README.md`
- `docs/method.md`
- `docs/reproducibility.md`
- `docs/references.md`
- `src/data_config.py`
- `src/main_LiteCDNET.py`
- `src/main_ablation.py`
- `src/ablation/`

这些部分主要负责：

- 公开仓库结构收敛
- 数据路径配置去本机化
- LiteCDNet 主入口和消融入口整理
- 面向外部读者的文档补全

## 2. 项目核心实现

以下部分与本项目主实验流程关系最紧密：

- `src/compare/LiteCDNET.py`
- `src/models/networks.py`
- `src/models/trainer.py`
- `src/models/evaluator.py`
- `src/models/losses.py`

其中既包含本项目的主模型实现，也包含将多个基线模型接入统一训练框架所需的工程化封装。尤其是 `src/models/networks.py`，既承担模型调度，也承载了部分项目内核心网络定义，例如 `SEIFNet`。

## 3. 对比模型与项目内适配

`src/compare/` 目录保留了多个用于基线对比的模型实现。这些文件对应的模型思想来源于各自论文或作者公开仓库，但在本仓库中通常还经过了下列类型的适配：

- 接入当前项目的 `define_G` 与统一训练入口
- 统一输入输出接口与损失调用方式
- 补齐或调整预训练权重加载逻辑
- 面向当前数据组织方式做工程化改造

因此，这些实现应理解为：

- “基于原论文/公开实现整理到本仓库训练框架中的实验版本”
- 而不是“全部从零独立原创的全新模型”

更细粒度的文件到论文/仓库映射，请查看 [references.md](references.md)。

## 4. 第三方辅助骨干与通用实现

部分辅助文件主要承担骨干网络或通用层实现，例如：

- `src/compare/MobileNet.py`
- `src/compare/resnet.py`
- `src/compare/resnet_tfi.py`
- `src/compare/resbase.py`

这些文件通常与公开深度学习实现、经典 backbone 结构或具体模型仓库中的辅助模块有关，主要目的在于支持统一实验而非作为本仓库单独的研究贡献点。

## 5. 建议的引用边界

如果你在阅读、复现或二次整理本仓库时需要说明来源，建议按下面的边界来写：

- 引用整个公开整理版仓库时，引用本仓库本身
- 使用某个具体对比模型时，同时引用该模型对应的原始论文或官方仓库
- 使用 LiteCDNet / SEIFNet / 消融实验组织方式时，可说明它们来自本项目公开整理版实现

简化表述可以写成：

- 本仓库是作者基于本科毕设实验代码整理出的公开复现版
- 对比模型部分基于对应论文或公开实现做了项目内适配
- 公开版新增了仓库结构收敛、数据路径配置和复现实验文档

## 6. 配套文件

- 更细粒度文件级映射：见 [references.md](references.md)
- 第三方代码边界摘要：见 [../NOTICE.md](../NOTICE.md)
- 仓库级引用入口：见 [../CITATION.cff](../CITATION.cff)
