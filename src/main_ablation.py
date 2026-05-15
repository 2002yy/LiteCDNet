from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

import torch

import utils_
from ablation.complexity import compare_with_full
from ablation.presets import (
    ABLATION_PRESETS,
    PAPER_DEFAULT_BOUNDARY_WEIGHT,
    PAPER_DEFAULT_LOSS_WEIGHTS,
    get_ablation_config,
)
from ablation.runner import train_and_eval
from utils_ import str2bool


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--ablation_case", default="full", choices=sorted(ABLATION_PRESETS.keys()), type=str)
    parser.add_argument("--gpu_ids", type=str, default="0")

    parser.add_argument("--checkpoint_root", default="checkpoints_ablation", type=str)
    parser.add_argument("--vis_root", default="vis_ablation", type=str)

    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dataset", default="CDDataset", type=str)
    parser.add_argument("--data_name", default="LEVIR", type=str)
    parser.add_argument("--data_root", default=None, type=str)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--split_val", default="val", type=str)
    parser.add_argument("--img_size", default=256, type=int)

    parser.add_argument("--n_class", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_epochs", default=150, type=int)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--use_amp", default=False, type=str2bool)
    parser.add_argument("--amp_dtype", default="fp16", choices=["fp16", "bf16"], type=str)
    parser.add_argument("--cache_clear_interval", default=10, type=int)
    parser.add_argument("--log_memory", default=True, type=str2bool)

    parser.add_argument("--deep_supervision", default=None, type=str)
    parser.add_argument(
        "--boundary_weight",
        "--L_boundary",
        dest="boundary_weight",
        default=None,
        type=float,
        help=f"Boundary loss weight. Thesis default is {PAPER_DEFAULT_BOUNDARY_WEIGHT}.",
    )
    parser.add_argument(
        "--loss_weights",
        nargs="+",
        type=float,
        default=None,
        help=f"Multi-scale supervision weights. Thesis default is: {' '.join(map(str, PAPER_DEFAULT_LOSS_WEIGHTS))}",
    )
    parser.add_argument("--loss_weight_1", default=None, type=float, help="Weight for the highest-resolution prediction.")
    parser.add_argument("--loss_weight_2", default=None, type=float, help="Weight for the second-scale prediction.")
    parser.add_argument("--loss_weight_3", default=None, type=float, help="Weight for the third-scale prediction.")
    parser.add_argument("--loss_weight_4", default=None, type=float, help="Weight for the fourth-scale prediction.")
    parser.add_argument("--fusion_mode", default=None, type=str)
    parser.add_argument("--context_mode", default=None, type=str)
    parser.add_argument("--decoder_mode", default=None, type=str)

    parser.add_argument("--run_test_only", default=False, type=str2bool)
    parser.add_argument("--project_name", default="", type=str)
    return parser


def _normalize_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y"}


def _resolve_loss_weights(args, preset: dict) -> list[float]:
    default_weights = list(preset["loss_weights"])

    if args.loss_weights is not None:
        if len(args.loss_weights) != 4:
            raise ValueError(
                f"--loss_weights expects exactly 4 values, got {len(args.loss_weights)}: {args.loss_weights}"
            )
        resolved = [float(x) for x in args.loss_weights]
    else:
        resolved = default_weights

    per_scale = [
        args.loss_weight_1,
        args.loss_weight_2,
        args.loss_weight_3,
        args.loss_weight_4,
    ]
    for idx, value in enumerate(per_scale):
        if value is not None:
            resolved[idx] = float(value)

    return resolved


def apply_case_defaults(args):
    preset = get_ablation_config(args.ablation_case)

    args.fusion_mode = args.fusion_mode or preset["fusion_mode"]
    args.context_mode = args.context_mode or preset["context_mode"]
    args.decoder_mode = args.decoder_mode or preset["decoder_mode"]
    args.deep_supervision = preset["deep_supervision"] if args.deep_supervision is None else _normalize_bool(args.deep_supervision)
    args.boundary_weight = preset["boundary_weight"] if args.boundary_weight is None else args.boundary_weight
    args.loss_weights = _resolve_loss_weights(args, preset)
    args.ablation_description = preset["description"]

    if not args.project_name:
        args.project_name = f"{args.data_name}_LiteCDNetAblation_{args.ablation_case}"

    return args


def finalize_paths(args):
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(",") if int(x) >= 0]
    utils_.get_device(args)
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    return args


def _running_under_debugger() -> bool:
    if sys.gettrace() is not None:
        return True
    debugger_modules = {"debugpy", "pydevd", "pydevd_file_utils"}
    if debugger_modules.intersection(sys.modules):
        return True
    return any(name in os.environ for name in ["VSCODE_PID", "PYDEVD_USE_FRAME_EVAL"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args()
    user_set_num_workers = "--num_workers" in sys.argv
    if _running_under_debugger() and not user_set_num_workers and args.num_workers > 0:
        print(
            f"[debug-safe] Debugger detected, auto-switching num_workers from {args.num_workers} to 0 "
            "to avoid Windows DataLoader spawn crashes.",
            flush=True,
        )
        args.num_workers = 0
    args = apply_case_defaults(args)
    args = finalize_paths(args)
    last_ckpt = os.path.join(args.checkpoint_dir, "last_ckpt.pt")
    best_ckpt = os.path.join(args.checkpoint_dir, "best_ckpt.pt")

    print(torch.cuda.is_available(), flush=True)
    print("=" * 60, flush=True)
    print(f"Running standalone ablation: {args.ablation_case}", flush=True)
    print(f"Description: {args.ablation_description}", flush=True)
    print(f"Project name: {args.project_name}", flush=True)
    print(f"Checkpoint dir: {args.checkpoint_dir}", flush=True)
    print(f"Vis dir: {args.vis_dir}", flush=True)
    print(f"Batch size: {args.batch_size} | Max epochs: {args.max_epochs} | GPU IDs: {args.gpu_ids}", flush=True)
    print(
        "Active loss config: "
        f"L_boundary={args.boundary_weight} | "
        f"loss_weights={args.loss_weights}",
        flush=True,
    )
    if (
        float(args.boundary_weight) != float(PAPER_DEFAULT_BOUNDARY_WEIGHT)
        or list(args.loss_weights) != list(PAPER_DEFAULT_LOSS_WEIGHTS)
    ):
        print(
            "Baseline paper reference: "
            f"L_boundary={PAPER_DEFAULT_BOUNDARY_WEIGHT} | "
            f"loss_weights={PAPER_DEFAULT_LOSS_WEIGHTS}",
            flush=True,
        )
    print("Preparing model complexity profiling...", flush=True)
    complexity = compare_with_full(args.ablation_case, img_size=args.img_size, n_class=args.n_class)
    print(
        "Complexity: "
        f"Params={complexity['params_m']:.2f}M | FLOPs={complexity['flops_g']:.2f}G | "
        f"vs_full dParams={complexity['delta_params_m_vs_full']:+.2f}M ({complexity['params_reduction_pct_vs_full']:+.2f}%) | "
        f"dFLOPs={complexity['delta_flops_g_vs_full']:+.2f}G ({complexity['flops_reduction_pct_vs_full']:+.2f}%)",
        flush=True,
    )
    print(f"Resume candidate: {'YES' if os.path.exists(last_ckpt) else 'NO'} | last_ckpt={last_ckpt}", flush=True)
    print(f"Best checkpoint exists: {'YES' if os.path.exists(best_ckpt) else 'NO'}", flush=True)
    print("=" * 60, flush=True)

    if args.run_test_only:
        from ablation.runner import AblationEvaluator

        print("Mode: evaluation only", flush=True)
        test_loader = utils_.get_loader(
            args.data_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            is_train=False,
            split="test",
            num_workers=args.num_workers,
            data_root=args.data_root,
        )
        evaluator = AblationEvaluator(args=args, dataloader=test_loader)
        print(evaluator.evaluate(), flush=True)
        return

    print("Mode: train + evaluation", flush=True)
    train_and_eval(args)


if __name__ == "__main__":
    main()
