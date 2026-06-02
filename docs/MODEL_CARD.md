# LiteCDNet Model Card

## Task

Remote sensing change detection on bi-temporal image pairs.

## Model

LiteCDNet uses a shared MobileNetV2 encoder, DiffFusion, LiteContextModule, add-based decoder, SE attention and multi-scale deep supervision.

## Dataset

LEVIR-CD is used as the main benchmark. Dataset is not redistributed in this repository.

## Metrics

Accuracy, mIoU, mF1, IoU(change), F1(change), Precision(change), Recall(change).

## Main Result

LiteCDNet reaches competitive LEVIR-CD accuracy with about 2.47M parameters and 2.14G FLOPs.

## Intended Use

Academic reproduction, lightweight change detection experiments, ablation study reference.

## Limitations

- No checkpoint is redistributed.
- Results depend on dataset split and training environment.
- The repository is for research and portfolio demonstration, not production deployment.

## Architecture Summary

- Shared MobileNetV2 encoder
- DiffFusion
- LiteContextModule
- Add-based decoder
- Multi-scale deep supervision
- CE + Dice + Boundary Loss

## Reproduction Notes

- Dataset should be prepared manually.
- Checkpoints are not redistributed.
- Main scripts are listed in README and docs/reproducibility.md.

## Portfolio Value

This repository demonstrates PyTorch training/evaluation organization, ablation design, metric reporting and public reproducibility cleanup.
