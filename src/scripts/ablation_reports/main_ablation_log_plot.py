from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from scripts.ablation_reports.main_ablation_log_extract import extract_logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize epoch-wise mF1/mIoU/Acc curves for LiteCDNet ablation logs."
    )
    parser.add_argument("--checkpoint_root", default="checkpoints_ablation", type=str)
    parser.add_argument("--extract_dir", default="ablation_reports/log_extract", type=str)
    parser.add_argument("--output_dir", default="ablation_reports/log_extract/plots", type=str)
    parser.add_argument("--projects", nargs="*", default=None)
    parser.add_argument("--only_completed", action="store_true", default=False)
    parser.add_argument("--skip_extract", action="store_true", default=False)
    return parser


def setup_matplotlib() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 140
    matplotlib.rcParams["savefig.dpi"] = 200


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    return float(v) if v not in {"", None} else float("nan")


def to_int(v: str) -> int:
    return int(v) if v not in {"", None} else -1


def slugify(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def is_completed(row: dict[str, object]) -> bool:
    return bool(str(row["completed"]).lower() == "true" or row["completed"] is True)


def plot_single_case(case_summary: dict[str, object], metrics_csv: Path, output_path: Path) -> None:
    rows = read_csv_rows(metrics_csv)
    train_rows = [r for r in rows if r["phase"] == "train"]
    val_rows = [r for r in rows if r["phase"] == "val"]

    train_epochs = [to_int(r["epoch"]) for r in train_rows]
    val_epochs = [to_int(r["epoch"]) for r in val_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor="#faf8f3")
    panels = [
        ("mf1", "mF1"),
        ("miou", "mIoU"),
        ("acc", "Accuracy"),
    ]
    train_color = "#2f6f7e"
    val_color = "#d97a2b"

    for ax, (metric_key, metric_label) in zip(axes, panels):
        train_vals = [to_float(r.get(metric_key, "")) for r in train_rows]
        val_vals = [to_float(r.get(metric_key, "")) for r in val_rows]
        if train_vals:
            ax.plot(train_epochs, train_vals, color=train_color, linewidth=2.0, label=f"Train {metric_label}")
        if val_vals:
            ax.plot(val_epochs, val_vals, color=val_color, linewidth=2.2, label=f"Val {metric_label}")
            best_idx = max(range(len(val_vals)), key=lambda i: val_vals[i])
            ax.scatter([val_epochs[best_idx]], [val_vals[best_idx]], color=val_color, s=35, zorder=3)
            ax.annotate(
                f"best={val_vals[best_idx]:.4f}@{val_epochs[best_idx]}",
                (val_epochs[best_idx], val_vals[best_idx]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
                color=val_color,
            )
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)

    status_text = "Completed" if is_completed(case_summary) else "In Progress"
    fig.suptitle(
        f"{case_summary['case_name']} | {status_text} | "
        f"best val mF1={case_summary['best_val_mf1']} | best val mIoU={case_summary['best_val_miou']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_completed_comparison(summaries: list[dict[str, object]], curves_dir: Path, output_path: Path) -> None:
    completed = [row for row in summaries if is_completed(row)]
    if not completed:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), facecolor="#fbfaf6")
    colors = ["#2f6f7e", "#d97a2b", "#5a8f3d", "#8f5a99", "#a94f4f", "#3a7ca5"]

    for idx, row in enumerate(completed):
        metrics_csv = curves_dir / f"{row['project_dir']}_epoch_metrics.csv"
        if not metrics_csv.exists():
            continue
        records = read_csv_rows(metrics_csv)
        val_rows = [r for r in records if r["phase"] == "val"]
        epochs = [to_int(r["epoch"]) for r in val_rows]
        val_mf1 = [to_float(r["mf1"]) for r in val_rows]
        val_miou = [to_float(r["miou"]) for r in val_rows]
        color = colors[idx % len(colors)]
        axes[0].plot(epochs, val_mf1, linewidth=2.1, color=color, label=str(row["case_name"]))
        axes[1].plot(epochs, val_miou, linewidth=2.1, color=color, label=str(row["case_name"]))

    axes[0].set_title("Validation mF1 Comparison", fontsize=12, fontweight="bold")
    axes[1].set_title("Validation mIoU Comparison", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Completed Ablation Runs: Validation Curve Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Ablation Curve Figure Notes",
        "",
        "## 综合图",
        "",
        "- 图题：消融实验验证集 mF1 与 mIoU 随 epoch 变化对比图",
        "- 英文图题：Comparison of validation mF1 and mIoU curves across completed ablation runs",
        "- 建议正文引用：如图所示，不同消融组在训练后期呈现出不同的收敛速度与性能上限，其中高性能组在验证集上的 mF1 与 mIoU 曲线整体更平稳，最佳值也更高。",
        "",
        "## 单组曲线图",
        "",
    ]
    for row in summaries:
        lines.extend(
            [
                f"- {row['case_name']}",
                f"  图题：{row['case_name']} 训练过程指标变化曲线",
                f"  英文图题：Training curves of {row['case_name']}",
                f"  建议正文引用：该图展示了 {row['case_name']} 在训练与验证阶段的 mF1、mIoU 与 Accuracy 变化趋势，可用于分析该结构调整对收敛行为和最终性能的影响。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    setup_matplotlib()

    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)
    curves_dir = extract_dir / "curves"

    if args.skip_extract:
        summaries = read_csv_rows(extract_dir / "ablation_log_summary.csv")
    else:
        summaries, _, _ = extract_logs(
            checkpoint_root=args.checkpoint_root,
            output_dir=extract_dir,
            projects=args.projects,
            only_completed=args.only_completed,
        )

    for row in summaries:
        if args.only_completed and not is_completed(row):
            continue
        metrics_csv = curves_dir / f"{row['project_dir']}_epoch_metrics.csv"
        if not metrics_csv.exists():
            continue
        output_path = output_dir / f"{slugify(str(row['project_dir']))}_curves.png"
        plot_single_case(row, metrics_csv, output_path)

    plot_completed_comparison(summaries, curves_dir, output_dir / "completed_val_mf1_miou_comparison.png")
    write_notes(output_dir / "figure_notes.md", summaries)

    print(f"generated_case_figures={len(list(output_dir.glob('*_curves.png')))}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
