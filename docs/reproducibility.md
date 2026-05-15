# 复现实验说明

## 1. Python 环境

建议：

- Python 3.10 或 3.11
- 先安装与你设备匹配的 PyTorch
- 再执行 `pip install -r requirements.txt`

## 2. 数据目录

默认情况下，脚本会在仓库根目录下查找：

```text
data/LEVIR
data/DSIFN_256
data/SYSU-CD
data/LEVIR-CD+_256
data/Big_Building_ChangeDetection
data/GZ
data/WHU-CUT
```

也可以使用：

- `--data_root <path>`
- `LITECDNET_DATA_ROOT`
- `LITECDNET_<DATASET>_ROOT`

例如：

```powershell
$env:LITECDNET_LEVIR_ROOT="D:\\datasets\\LEVIR"
python -X utf8 src/main_LiteCDNET.py --data_name LEVIR
```

## 3. 推荐复现实验入口

LiteCDNet 主实验：

```bash
python -X utf8 src/main_LiteCDNET.py --data_name LEVIR
```

SEIFNet 对比实验：

```bash
python -X utf8 src/main_train.py --net_G SEIFNet --data_name LEVIR
```

LiteCDNet 消融实验：

```bash
python -X utf8 src/main_ablation.py --ablation_case full --data_name LEVIR
python -X utf8 src/main_ablation.py --ablation_case no_context --data_name LEVIR
```

评估指定 checkpoint：

```bash
python -X utf8 src/eval_cd.py --project_name LEVIR_LiteCDNet_BCEDiceBoundary0.3_AdamW_Cosine_150 --checkpoint_name best_ckpt.pt --data_name LEVIR
```

## 4. 输出位置

训练与评估默认输出到：

- `checkpoints/`
- `checkpoints_ablation/`
- `vis/`
- `vis_ablation/`

这些目录均已加入忽略规则，不会进入公开仓库。

## 5. 复现注意事项

- `DMINet` 等部分对比模型可能需要额外的本地预训练权重
- 仓库不提供数据集和模型权重，只提供训练与复现代码
- 如果要做严格对照实验，请固定随机种子、PyTorch 版本和 CUDA 环境
