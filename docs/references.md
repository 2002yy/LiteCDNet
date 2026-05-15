# 细粒度论文与仓库引用清单

这个文档按“本仓库文件 -> 对应论文 / 官方仓库 / 适配说明”的粒度整理，方便公开仓库时更明确地区分项目内实现、第三方基线和统一训练框架适配部分。

说明：

- “适配状态”中的“项目内适配”表示本仓库文件不是简单镜像，而是被整理到当前训练框架中使用
- “本地变体”表示保留了原始方法思路，但文件本身已经是本项目演化出的版本
- 少数文件是项目内原创或 thesis-local 代码，没有单独外部仓库可以对应

## 1. 对比模型文件映射

| 本仓库文件 | 对应方法 / 论文 | 官方仓库 / 代码来源 | 适配状态 |
| --- | --- | --- | --- |
| `src/compare/FC_EF.py` | [Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462) | [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) | 项目内适配 |
| `src/compare/FC_Siam_conc.py` | [Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462) | [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) | 项目内适配 |
| `src/compare/FC_Siam_diff.py` | [Fully Convolutional Siamese Networks for Change Detection](https://arxiv.org/abs/1810.08462) | [rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection) | 项目内适配 |
| `src/compare/NestedUNet.py` | [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) | 项目内适配，用作变化检测基线骨干 |
| `src/compare/SNUNet.py` | [SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images](https://ieeexplore.ieee.org/document/9355573) | [likyoo/Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet) | 项目内适配 |
| `src/compare/IFNet.py` | [A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images](https://www.sciencedirect.com/science/article/pii/S0924271620301532) | [GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) | 项目内适配 |
| `src/compare/DASNet.py` | [DASNet: Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images](https://arxiv.org/abs/2003.03608) | [lehaifeng/DASNet](https://github.com/lehaifeng/DASNet) | 项目内适配 |
| `src/compare/DTCDSCN.py` | [Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model](https://arxiv.org/abs/1909.07726) | [fitzpchao/DTCDSCN](https://github.com/fitzpchao/DTCDSCN) | 项目内适配 |
| `src/compare/ChangeFormer.py` | [A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293) | [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) | 项目内适配 |
| `src/compare/A2Net.py` | [Lightweight Remote Sensing Change Detection with Progressive Aggregation and Supervised Attention](https://ieeexplore.ieee.org/document/10129120) | [guanyuezhen/A2Net](https://github.com/guanyuezhen/A2Net) | 项目内适配 |
| `src/compare/A2Net_v2.py` | A2Net 思路的项目内扩展版本 | [guanyuezhen/A2Net](https://github.com/guanyuezhen/A2Net) | 本地变体，不应视为官方 A2Net 原样实现 |
| `src/compare/TFI_GR.py` | [Remote Sensing Change Detection via Temporal Feature Interaction and Guided Refinement](https://doi.org/10.1109/TGRS.2022.3199502) | [guanyuezhen/TFI-GR](https://github.com/guanyuezhen/TFI-GR) | 项目内适配 |
| `src/compare/DMINet.py` | [Change Detection on Remote Sensing Images using Dual-branch Multi-level Inter-temporal Network](https://doi.org/10.1109/TGRS.2023.3241257) | [ZhengJianwei2/DMINet](https://github.com/ZhengJianwei2/DMINet) | 项目内适配 |
| `src/compare/Changer.py` | [Changer: Feature Interaction is What You Need for Change Detection](https://arxiv.org/abs/2209.08290) | [likyoo/open-cd](https://github.com/likyoo/open-cd) | 保留为对比/扩展参考，当前仓库默认入口未重点维护 |
| `src/compare/LiteCDNET.py` | LiteCDNet 项目主模型实现 | 无独立外部官方仓库；属本项目公开版核心实现 | 项目内原创 / thesis-local |

## 2. 骨干与辅助文件映射

| 本仓库文件 | 对应来源 | 官方仓库 / 论文 | 说明 |
| --- | --- | --- | --- |
| `src/compare/MobileNet.py` | MobileNetV2 / torchvision 相关实现 | [pytorch/vision](https://github.com/pytorch/vision) | 用于部分轻量对比模型骨干或权重兼容 |
| `src/compare/resnet.py` | ResNet 系列基础实现 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) / [pytorch/vision](https://github.com/pytorch/vision) | 通用 backbone 支撑文件 |
| `src/compare/resnet_tfi.py` | TFI-GR 配套 ResNet 辅助实现 | [guanyuezhen/TFI-GR](https://github.com/guanyuezhen/TFI-GR) | 随 TFI-GR 一起接入当前仓库 |
| `src/compare/resbase.py` | DASNet 相关骨干辅助实现 | [lehaifeng/DASNet](https://github.com/lehaifeng/DASNet) | 服务于 DASNet 运行 |

## 3. 项目内核心组织文件

以下文件更适合理解为“本项目公开整理层”，而不是单独对应某一篇外部论文：

- `src/data_config.py`
- `src/main_LiteCDNET.py`
- `src/main_train.py`
- `src/main_ablation.py`
- `src/models/networks.py`
- `src/models/trainer.py`
- `src/models/evaluator.py`
- `src/ablation/`

其中：

- `src/models/networks.py` 同时承担模型调度和 `SEIFNet` 等项目内核心实现
- `src/ablation/` 是 LiteCDNet 消融实验在本仓库下的独立整理结果
- `src/data_config.py` 是公开版为去除本机绝对路径依赖而新增的路径解析层

## 4. 使用建议

如果你只引用整个公开仓库：

- 请优先引用仓库本身，见 [../CITATION.cff](../CITATION.cff)

如果你使用了某个具体对比模型：

- 请同时引用该模型对应的原论文或官方仓库
- 不要把本仓库里的适配版误写成该方法唯一或官方实现

如果你要在论文、报告或 README 中做简短说明，可以使用下面这类表述：

- “本仓库在统一训练框架下整理并接入了多个公开变化检测模型用于基线比较，细粒度来源见 `docs/references.md`。”
- “A2Net_v2 为本项目基于 A2Net 思路演化出的本地变体，不等同于官方 A2Net 仓库原始实现。”
