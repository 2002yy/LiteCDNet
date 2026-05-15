# Release Notes: v1.0.0

建议版本号：

- `v1.0.0`

建议发布日期：

- LiteCDNet 首个公开整理版发布日

## 中文版

### 概述

`v1.0.0` 是 LiteCDNet 毕设代码仓库的首个公开整理版。这个版本从原始实验工作区中收敛而来，重点是把训练、评估、消融和基线对比相关代码整理成一个可公开查看、可继续维护、便于复现的仓库结构。

### 本次发布包含

- 公开版目录重组，统一为 `src/`、`docs/`、`assets/`
- 保留 LiteCDNet 主模型训练与评估入口
- 保留 SEIFNet 与多个对比模型的统一训练入口
- 保留 LiteCDNet A0-A7 消融实验相关代码
- 新增公开版 README、文档索引、复现实验说明和方法说明
- 新增数据路径配置层，移除作者本机绝对路径依赖
- 新增来源说明、细粒度论文/仓库引用清单、`NOTICE` 与 `CITATION.cff`

### 本次发布不包含

- 数据集
- checkpoint 权重
- 大量可视化输出
- 论文写作草稿、答辩材料和个人工作目录

### 重要说明

- `src/compare/` 中的部分模型文件属于“基于论文或官方实现整理到统一训练框架中的项目内适配版本”
- `A2Net_v2` 为项目内演化版本，不应等同视为官方 A2Net 原始实现
- `DMINet` 等少数模型可能需要额外的本地预训练权重文件

### 面向公开读者的推荐入口

- 仓库概览：`README.md`
- 双语简介：`docs/project-overview-bilingual.md`
- 复现说明：`docs/reproducibility.md`
- 来源说明：`docs/attribution.md`
- 文件级引用清单：`docs/references.md`

### 已知边界

- 当前发布重点是仓库整理与公开边界清晰化，不代表所有模型都已在当前环境完成重新训练验证
- 仓库不自带数据与权重，因此第一次运行前仍需自行准备数据集与部分外部依赖

## English Version

### Overview

`v1.0.0` is the first public, cleaned-up release of the LiteCDNet graduation-project repository. This release narrows the original working directory into a more maintainable and reproducible public repository focused on training, evaluation, ablation, and baseline comparison.

### Included in this release

- Public-facing repository restructuring into `src/`, `docs/`, and `assets/`
- Preserved entry points for LiteCDNet training and evaluation
- Preserved unified training entry points for SEIFNet and multiple baseline models
- Preserved code for LiteCDNet A0-A7 ablation experiments
- Added a public README, documentation index, reproducibility guide, and method summary
- Added configurable dataset-path resolution to remove machine-specific absolute paths
- Added repository-level attribution, file-level references, `NOTICE`, and `CITATION.cff`

### Not included in this release

- Datasets
- Trained checkpoints
- Large visualization outputs
- Thesis drafting materials, defense slides, or personal workspace mirrors

### Important notes

- Some files under `src/compare/` should be understood as project-internal adapted versions of published or publicly released models under a unified training framework
- `A2Net_v2` is a project-local evolution based on A2Net-style ideas rather than the official upstream implementation
- Some models, such as `DMINet`, may still require additional local pretrained weights

### Recommended entry points for public readers

- Repository overview: `README.md`
- Bilingual project page: `docs/project-overview-bilingual.md`
- Reproducibility guide: `docs/reproducibility.md`
- Attribution summary: `docs/attribution.md`
- File-level reference list: `docs/references.md`

### Known boundaries

- This release primarily focuses on repository cleanup and public-facing documentation rather than full re-training verification of every included model
- Since datasets and checkpoints are not distributed in this repository, users still need to prepare external data and some model dependencies before running experiments
