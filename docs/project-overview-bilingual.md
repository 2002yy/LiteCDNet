# LiteCDNet Project Overview / LiteCDNet 项目简介

## Chinese

### 项目定位

LiteCDNet 是一个面向遥感变化检测实验的公开整理版仓库，来源于作者本科毕业设计期间的实验代码收敛与工程化清理。公开版的目标不是完整保留原始工作区，而是把真正适合公开展示、继续复现和后续维护的部分整理出来。

### 仓库目标

这个仓库主要服务于三类需求：

- 展示 LiteCDNet 及相关实验组织方式
- 复现 LiteCDNet、SEIFNet 和若干对比模型的训练与评估流程
- 为后续论文写作、项目答辩展示和仓库公开提供更清晰的来源边界

### 当前公开内容

公开版目前保留：

- LiteCDNet 主模型实现与训练入口
- SEIFNet 与多种对比模型的统一训练入口
- LiteCDNet A0-A7 消融实验相关代码
- 环境、数据、训练、评估和引用说明文档

### 当前不公开内容

为保证仓库体积、公开边界和个人工作区隐私，以下内容不在公开版中分发：

- 数据集原始文件
- checkpoint 权重文件
- 大量可视化结果与日志产物
- 论文草稿、答辩材料和本地工作目录镜像

### 代码来源边界

本仓库并不是由单一来源代码组成。

- 一部分代码属于本项目核心实现与公开整理内容
- 一部分代码用于基线对比，来源于已有论文思路或公开仓库
- 这些对比模型在当前仓库中通常经过了统一接口、路径和训练流程的项目内适配

因此，阅读或引用时，建议同时参考：

- 仓库级来源说明：`docs/attribution.md`
- 文件级论文 / 仓库映射：`docs/references.md`

### 适合谁阅读

这个仓库尤其适合以下读者：

- 想了解 LiteCDNet 毕设项目公开版组织方式的人
- 想复现遥感变化检测训练流程的人
- 想查看某些常见变化检测基线如何接入统一训练框架的人

### 使用建议

如果你是第一次进入仓库，建议按以下顺序阅读：

1. `README.md`
2. `docs/project-overview-bilingual.md`
3. `docs/reproducibility.md`
4. `docs/attribution.md`
5. `docs/references.md`

## English

### Project Positioning

LiteCDNet is a public, cleaned-up repository for remote sensing change detection experiments. It was derived from the author's undergraduate graduation-project workspace through repository narrowing and engineering cleanup. The goal of this public version is not to preserve the entire original workspace, but to keep the parts that are suitable for public presentation, reproducibility, and ongoing maintenance.

### Repository Goals

This repository mainly serves three purposes:

- presenting LiteCDNet and its associated experiment organization
- reproducing the training and evaluation workflows of LiteCDNet, SEIFNet, and several baseline models
- clarifying provenance boundaries for thesis writing, project presentation, and public repository publishing

### What is currently included

The public release currently preserves:

- the main LiteCDNet implementation and training entry points
- unified training entry points for SEIFNet and multiple baseline models
- code related to LiteCDNet A0-A7 ablation experiments
- documentation for environment setup, data preparation, training, evaluation, and attribution

### What is intentionally excluded

To keep the repository lightweight, publicly safe, and detached from the personal workspace, the following items are not distributed:

- raw datasets
- trained checkpoints
- large visualization outputs and log artifacts
- thesis drafts, defense materials, and mirrored local workspace directories

### Provenance boundaries

This repository is not composed of code from a single source.

- Some parts are core project implementations and public-release engineering work
- Some parts are preserved for baseline comparison and derive from published papers or public repositories
- Those baseline files are often adapted inside this repository to fit a unified interface, path convention, and training workflow

Readers and users are therefore encouraged to consult both:

- repository-level attribution: `docs/attribution.md`
- file-level paper / repository mapping: `docs/references.md`

### Intended readers

This repository is especially useful for readers who want to:

- understand the public-facing organization of the LiteCDNet graduation project
- reproduce remote sensing change detection training workflows
- inspect how common baseline models are integrated into a unified training framework

### Suggested reading order

If you are new to this repository, a practical reading order is:

1. `README.md`
2. `docs/project-overview-bilingual.md`
3. `docs/reproducibility.md`
4. `docs/attribution.md`
5. `docs/references.md`
