# 方法说明

## 1. 任务背景

LiteCDNet 面向双时相遥感影像变化检测任务。输入为同一区域在两个时间点的影像，输出为像素级变化掩码。

## 2. LiteCDNet 的公开版关注点

公开版仓库主要保留以下研究实现：

- 轻量级主干与双时相特征提取
- 差异特征融合模块
- 上下文增强模块
- 解码恢复与多尺度监督
- 与 SEIFNet、A2Net、IFNet、ChangeFormer 等方法的统一对比入口

## 3. 代码对应关系

- `src/main_LiteCDNET.py`：LiteCDNet 主训练入口
- `src/models/networks.py`：模型装配与 `define_G`
- `src/compare/LiteCDNET.py`：LiteCDNet 网络实现
- `src/ablation/`：A0-A7 消融实验实现
- `src/eval_cd.py`：统一评估入口

## 4. 公开版与原始工作区的差异

为便于对外共享与复现，公开版做了以下收敛：

- 移除了 checkpoint、vis、日志和缓存文件
- 移除了答辩材料、论文排版脚本和个人目录镜像
- 将数据路径改成仓库相对路径或显式可配置路径
- 将仓库结构整理为 `src/`、`docs/` 和 `assets/`
