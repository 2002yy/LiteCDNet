from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.complexity import compare_with_full

CASE_ORDER = [
    "full",
    "abs_diff",
    "no_context",
    "concat_decoder",
    "no_boundary",
    "no_deep_supervision",
    "boundary_05",
    "loss_weights_080604",
    "c45_diff_absconcat",
    "c45_context_triple",
]

CASE_META = {
    "full": {"code": "A0", "label_cn": "完整 LiteCDNet", "feature_cn": "基线结构"},
    "abs_diff": {"code": "A1", "label_cn": "固定绝对差分", "feature_cn": "去除可学习差分"},
    "no_context": {"code": "A2", "label_cn": "去除 LiteContext", "feature_cn": "去除上下文增强"},
    "concat_decoder": {"code": "A3", "label_cn": "拼接式解码", "feature_cn": "重型解码对照"},
    "no_boundary": {"code": "A4", "label_cn": "去除边界损失", "feature_cn": "训练策略变化"},
    "no_deep_supervision": {"code": "A5", "label_cn": "去除深监督", "feature_cn": "训练策略变化"},
    "boundary_05": {"code": "A6", "label_cn": "边界权重=0.5", "feature_cn": "训练权重调节"},
    "loss_weights_080604": {"code": "A7", "label_cn": "监督权重=1.0/0.8/0.6/0.4", "feature_cn": "多尺度监督权重调整"},
    "c45_diff_absconcat": {"code": "E1", "label_cn": "增强 C4/C5 差分融合", "feature_cn": "高层定点增强"},
    "c45_context_triple": {"code": "E2", "label_cn": "增强 C4/C5 三分支上下文", "feature_cn": "高层定点增强"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Params/FLOPs reports and figures for LiteCDNet ablation cases.")
    parser.add_argument("--output_dir", type=str, default="", help="Optional custom output directory")
    parser.add_argument("--metadata_dir", type=str, default="", help="Optional custom metadata directory")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size for complexity profiling")
    parser.add_argument("--n_class", type=int, default=2, help="Number of output classes")
    parser.add_argument("--cases", nargs="+", default=None, help="Optional subset of cases to include")
    parser.add_argument("--ymin_params", type=float, default=None, help="Optional lower bound for Params axis")
    parser.add_argument("--ymin_flops", type=float, default=None, help="Optional lower bound for FLOPs axis")
    return parser


def set_plot_style() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def build_rows(img_size: int, n_class: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in CASE_ORDER:
        meta = CASE_META[case]
        complexity = compare_with_full(case, img_size=img_size, n_class=n_class)
        rows.append(
            {
                "case": case,
                "code": meta["code"],
                "label_cn": meta["label_cn"],
                "feature_cn": meta["feature_cn"],
                "params_m": round(complexity["params_m"], 6),
                "macs_g": round(complexity["macs_g"], 6),
                "flops_g": round(complexity["flops_g"], 6),
                "delta_params_m_vs_full": round(complexity["delta_params_m_vs_full"], 6),
                "delta_flops_g_vs_full": round(complexity["delta_flops_g_vs_full"], 6),
                "params_reduction_pct_vs_full": round(complexity["params_reduction_pct_vs_full"], 6),
                "flops_reduction_pct_vs_full": round(complexity["flops_reduction_pct_vs_full"], 6),
            }
        )
    return rows


def filter_rows(rows: list[dict[str, object]], selected_cases: list[str] | None) -> list[dict[str, object]]:
    if not selected_cases:
        return rows
    selected = set(selected_cases)
    return [row for row in rows if str(row["case"]) in selected]


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# LiteCDNet 各实验组 Params/FLOPs 统计",
        "",
        "| 编号 | 配置 | Params/M | FLOPs/G | 相对 A0 Params 变化/M | 相对 A0 FLOPs 变化/G | Params 降幅/% | FLOPs 降幅/% | 结构特点 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["code"]),
                    str(row["label_cn"]),
                    f"{float(row['params_m']):.4f}",
                    f"{float(row['flops_g']):.4f}",
                    f"{float(row['delta_params_m_vs_full']):+.4f}",
                    f"{float(row['delta_flops_g_vs_full']):+.4f}",
                    f"{float(row['params_reduction_pct_vs_full']):+.2f}",
                    f"{float(row['flops_reduction_pct_vs_full']):+.2f}",
                    str(row["feature_cn"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_summary_text(rows: list[dict[str, object]]) -> str:
    lightest_params = min(rows, key=lambda x: float(x["params_m"]))
    lightest_flops = min(rows, key=lambda x: float(x["flops_g"]))
    heaviest_params = max(rows, key=lambda x: float(x["params_m"]))
    heaviest_flops = max(rows, key=lambda x: float(x["flops_g"]))

    lines = [
        "LiteCDNet 复杂度分析摘要",
        "",
        f"参数量最小的配置是 {lightest_params['code']} {lightest_params['label_cn']}，Params={float(lightest_params['params_m']):.4f}M。",
        f"FLOPs 最小的配置是 {lightest_flops['code']} {lightest_flops['label_cn']}，FLOPs={float(lightest_flops['flops_g']):.4f}G。",
        f"参数量最大的配置是 {heaviest_params['code']} {heaviest_params['label_cn']}，Params={float(heaviest_params['params_m']):.4f}M。",
        f"FLOPs 最大的配置是 {heaviest_flops['code']} {heaviest_flops['label_cn']}，FLOPs={float(heaviest_flops['flops_g']):.4f}G。",
        "",
        "关键观察：",
        "1. A1 和 A2 仍是最轻的两组，说明可学习差分融合和 LiteContext 的额外开销有限但可感知。",
        "2. A3 的 Params 与 FLOPs 明显升高，进一步说明拼接式解码比加法式解码更重。",
        "3. A4、A5、A6 与 A7 主要改变训练策略或损失权重，推理复杂度与 A0 基本一致。",
        "4. E1、E2 仅在高层 C4/C5 做定点增强，复杂度只小幅增加，适合作为 LiteCDNet 的轻量扩展方向。",
    ]
    return "\n".join(lines) + "\n"


def _apply_zoomed_ylim(ax, values: list[float], ymin: float | None) -> None:
    if ymin is None or not values:
        return
    ymax = max(values)
    upper_margin = max(0.03, (ymax - ymin) * 0.12)
    ax.set_ylim(ymin, ymax + upper_margin)


def create_overview_figure(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    ymin_params: float | None = None,
    ymin_flops: float | None = None,
) -> None:
    set_plot_style()
    codes = [str(row["code"]) for row in rows]
    x = list(range(len(rows)))
    params_vals = [float(row["params_m"]) for row in rows]
    flops_vals = [float(row["flops_g"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor("#faf8f3")

    ax1, ax2 = axes
    bars1 = ax1.bar(x, params_vals, color="#3f7f93")
    ax1.set_title("Params Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(codes)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    _apply_zoomed_ylim(ax1, params_vals, ymin_params)
    for bar, value in zip(bars1, params_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, flops_vals, color="#c65f46")
    ax2.set_title("FLOPs Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(codes)
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    _apply_zoomed_ylim(ax2, flops_vals, ymin_flops)
    for bar, value in zip(bars2, flops_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("LiteCDNet Complexity Report", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.02, "A0-A7 denote standard ablations. E1-E2 denote high-level enhancement settings.", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_structure_summary_figure(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    ymin_params: float | None = None,
    ymin_flops: float | None = None,
) -> None:
    set_plot_style()
    params_vals = [float(row["params_m"]) for row in rows]
    flops_vals = [float(row["flops_g"]) for row in rows]
    full_params = params_vals[0]
    full_flops = flops_vals[0]
    params_ratio = [value / full_params for value in params_vals]
    flops_ratio = [value / full_flops for value in flops_vals]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.55]},
        facecolor="#fbfaf7",
    )

    ax1, ax2, ax3 = axes
    x = list(range(len(rows)))
    codes = [row["code"] for row in rows]

    ax1.bar(x, params_vals, color="#4f7c82")
    ax1.set_title("Params Comparison", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(codes)
    ax1.set_ylabel("Params (M)")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    _apply_zoomed_ylim(ax1, params_vals, ymin_params)
    for xi, yi in zip(x, params_vals):
        ax1.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax2.bar(x, flops_vals, color="#c96b4b")
    ax2.set_title("FLOPs Comparison", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(codes)
    ax2.set_ylabel("FLOPs (G)")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    _apply_zoomed_ylim(ax2, flops_vals, ymin_flops)
    for xi, yi in zip(x, flops_vals):
        ax2.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax3.axis("off")
    ax3.set_title("Params-FLOPs-Structure Summary", fontsize=13, fontweight="bold", pad=10)
    table_rows = []
    for idx, row in enumerate(rows):
        table_rows.append(
            [
                row["code"],
                row["label_cn"],
                f"{row['params_m']:.2f}M / {row['flops_g']:.2f}G",
                f"{params_ratio[idx]:.3f}x / {flops_ratio[idx]:.3f}x",
                row["feature_cn"],
            ]
        )
    table = ax3.table(
        cellText=table_rows,
        colLabels=["编号", "配置", "Params/FLOPs", "相对 A0", "结构特点"],
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.24, 0.20, 0.18, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    fig.suptitle("LiteCDNet 各实验组 Params-FLOPs-结构特点综合信息图", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "A0-A7 为标准消融与训练策略组，E1-E2 为仅在 C4/C5 进行高层定点增强的补充扩展组。",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.98, hspace=0.45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_case_card(
    row: dict[str, object],
    full_row: dict[str, object],
    output_path: Path,
    *,
    ymin_params: float | None = None,
    ymin_flops: float | None = None,
) -> None:
    set_plot_style()
    labels = ["Params (M)", "FLOPs (G)"]
    full_vals = [float(full_row["params_m"]), float(full_row["flops_g"])]
    case_vals = [float(row["params_m"]), float(row["flops_g"])]
    x = [0, 1]
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor("#fcfbf8")
    bars1 = ax.bar([i - width / 2 for i in x], full_vals, width=width, color="#9db4c0", label="A0 Full")
    bars2 = ax.bar([i + width / 2 for i in x], case_vals, width=width, color="#d97b66", label=str(row["code"]))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{row['code']} {row['label_cn']} vs A0 Full", fontsize=13, fontweight="bold")
    combined_vals = full_vals + case_vals
    zoom_ymin = None
    if ymin_params is not None or ymin_flops is not None:
        candidate = [v for v in [ymin_params, ymin_flops] if v is not None]
        if candidate:
            zoom_ymin = min(candidate)
    _apply_zoomed_ylim(ax, combined_vals, zoom_ymin)

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    info_lines = [
        f"结构特点：{row['feature_cn']}",
        f"Params 变化：{float(row['delta_params_m_vs_full']):+.4f}M ({float(row['params_reduction_pct_vs_full']):+.2f}%)",
        f"FLOPs 变化：{float(row['delta_flops_g_vs_full']):+.4f}G ({float(row['flops_reduction_pct_vs_full']):+.2f}%)",
    ]
    fig.text(0.08, 0.02, "\n".join(info_lines), fontsize=10, va="bottom")
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolve_metadata_dir(output_dir: Path, metadata_dir_arg: str) -> Path:
    if metadata_dir_arg:
        return Path(metadata_dir_arg).resolve()

    image_root = (PROJECT_ROOT / "论文插图素材" / "可直接使用图_20260422").resolve()
    derived_root = (PROJECT_ROOT / "论文插图素材" / "_derived_metadata").resolve()

    try:
        relative = output_dir.relative_to(image_root)
    except ValueError:
        return (output_dir.parent / "_derived_metadata" / output_dir.name).resolve()

    return (derived_root / relative).resolve()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (PROJECT_ROOT / "ablation_reports" / "complexity_report").resolve()
    cards_dir = output_dir / "case_cards"
    metadata_dir = resolve_metadata_dir(output_dir, args.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(img_size=args.img_size, n_class=args.n_class)
    rows = filter_rows(rows, args.cases)
    full_row = next(row for row in rows if row["case"] == "full")

    write_csv(rows, metadata_dir / "all_cases_complexity.csv")
    write_json(rows, metadata_dir / "all_cases_complexity.json")
    (metadata_dir / "all_cases_complexity.md").write_text(build_markdown(rows), encoding="utf-8-sig")
    (metadata_dir / "complexity_analysis.txt").write_text(build_summary_text(rows), encoding="utf-8-sig")
    create_overview_figure(
        rows,
        output_dir / "complexity_overview.png",
        ymin_params=args.ymin_params,
        ymin_flops=args.ymin_flops,
    )
    create_overview_figure(
        rows,
        output_dir / "图4-x_复杂度总览图.png",
        ymin_params=args.ymin_params,
        ymin_flops=args.ymin_flops,
    )
    create_structure_summary_figure(
        rows,
        output_dir / "fig4-7_params_flops_structure_summary.png",
        ymin_params=args.ymin_params,
        ymin_flops=args.ymin_flops,
    )
    create_structure_summary_figure(
        rows,
        output_dir / "图4-7_Params-FLOPs-结构特点综合信息图.png",
        ymin_params=args.ymin_params,
        ymin_flops=args.ymin_flops,
    )

    for row in rows:
        create_case_card(
            row,
            full_row,
            cards_dir / f"{row['code']}_{row['case']}.png",
            ymin_params=args.ymin_params,
            ymin_flops=args.ymin_flops,
        )

    print(f"output_dir: {output_dir}")
    print(f"metadata_dir: {metadata_dir}")
    print("generated_files:")
    for path in [
        metadata_dir / "all_cases_complexity.csv",
        metadata_dir / "all_cases_complexity.json",
        metadata_dir / "all_cases_complexity.md",
        metadata_dir / "complexity_analysis.txt",
        output_dir / "complexity_overview.png",
        output_dir / "图4-x_复杂度总览图.png",
        output_dir / "fig4-7_params_flops_structure_summary.png",
        output_dir / "图4-7_Params-FLOPs-结构特点综合信息图.png",
    ]:
        print(f"- {path}")
    print(f"- {cards_dir}")


if __name__ == "__main__":
    main()
