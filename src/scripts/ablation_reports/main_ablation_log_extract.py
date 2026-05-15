from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EPOCH_LINE_RE = re.compile(
    r"Is_training:\s+(True|False)\.\s+Epoch\s+(\d+)\s+/\s+(\d+),\s+epoch_mF1=\s*([0-9.]+)"
)
METRIC_PAIR_RE = re.compile(r"([A-Za-z0-9_]+):\s*([0-9.]+)")
SUMMARY_OLD_RE = re.compile(
    r"\[epoch_summary\]\s+epoch=(\d+)\s+train_mF1=([0-9.]+)\s+val_mF1=([0-9.]+)\s+"
    r"epoch_time=([0-9.]+)\s+min\s+rough_remaining=([0-9a-zA-Z. ]+)\s+"
    r"best_epoch=(\d+)\s+best_val_mF1=([0-9.]+)"
)
SUMMARY_NEW_RE = re.compile(
    r"\[epoch_summary\]\s+epoch=(\d+)/(\d+)\s+\|\s+train_mF1=([0-9.]+)\s+\|\s+val_mF1=([0-9.]+)\s+\|"
    r"\s+epoch_time=([0-9.]+)\s+min\s+\|\s+remaining=([0-9a-zA-Z. ]+)\s+\|\s+AMP=(ON|OFF)(?:\(([^)]+)\))?\s+\|"
    r"\s+peak_allocated=([0-9.]+)MB,\s+peak_reserved=([0-9.]+)MB\s+\|\s+best_epoch=(\d+)\s+\|\s+best_val_mF1=([0-9.]+)"
)
PROJECT_NAME_RE = re.compile(r"project_name:\s+([^\s]+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract epoch-wise ablation metrics from log.txt and summarize completed runs."
    )
    parser.add_argument("--checkpoint_root", default="checkpoints_ablation", type=str)
    parser.add_argument("--output_dir", default="ablation_reports/log_extract", type=str)
    parser.add_argument("--projects", nargs="*", default=None, help="Optional checkpoint directory names to parse.")
    parser.add_argument("--only_completed", default=False, action="store_true")
    return parser


def infer_case_name(project_dir: str) -> str:
    prefix = "LEVIR_LiteCDNetAblation_"
    if project_dir.startswith(prefix):
        return project_dir[len(prefix):]
    return project_dir


def parse_metric_line(metric_line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in METRIC_PAIR_RE.findall(metric_line):
        metrics[key] = float(value)
    return metrics


def parse_log(log_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    epoch_records: list[dict[str, object]] = []
    epoch_summaries: list[dict[str, object]] = []
    project_name = log_path.parent.name

    pending_epoch: dict[str, object] | None = None

    for idx, line in enumerate(lines):
        if idx == 0:
            match = PROJECT_NAME_RE.search(line)
            if match:
                project_name = match.group(1)

        epoch_match = EPOCH_LINE_RE.search(line)
        if epoch_match:
            pending_epoch = {
                "phase": "train" if epoch_match.group(1) == "True" else "val",
                "epoch": int(epoch_match.group(2)),
                "max_epoch": int(epoch_match.group(3)),
                "epoch_mf1": float(epoch_match.group(4)),
            }
            continue

        if pending_epoch is not None and line.strip().startswith("acc:"):
            record = dict(pending_epoch)
            record.update(parse_metric_line(line))
            epoch_records.append(record)
            pending_epoch = None
            continue

        old_summary = SUMMARY_OLD_RE.search(line)
        if old_summary:
            epoch_summaries.append(
                {
                    "epoch": int(old_summary.group(1)),
                    "max_epoch": None,
                    "train_mf1": float(old_summary.group(2)),
                    "val_mf1": float(old_summary.group(3)),
                    "epoch_time_min": float(old_summary.group(4)),
                    "remaining": old_summary.group(5).strip(),
                    "amp_enabled": "",
                    "amp_dtype": "",
                    "peak_allocated_mb": None,
                    "peak_reserved_mb": None,
                    "best_epoch": int(old_summary.group(6)),
                    "best_val_mf1": float(old_summary.group(7)),
                }
            )
            continue

        new_summary = SUMMARY_NEW_RE.search(line)
        if new_summary:
            epoch_summaries.append(
                {
                    "epoch": int(new_summary.group(1)),
                    "max_epoch": int(new_summary.group(2)),
                    "train_mf1": float(new_summary.group(3)),
                    "val_mf1": float(new_summary.group(4)),
                    "epoch_time_min": float(new_summary.group(5)),
                    "remaining": new_summary.group(6).strip(),
                    "amp_enabled": new_summary.group(7),
                    "amp_dtype": new_summary.group(8) or "",
                    "peak_allocated_mb": float(new_summary.group(9)),
                    "peak_reserved_mb": float(new_summary.group(10)),
                    "best_epoch": int(new_summary.group(11)),
                    "best_val_mf1": float(new_summary.group(12)),
                }
            )

    return epoch_records, epoch_summaries, project_name


def summarize_project(project_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    log_path = project_dir / "log.txt"
    epoch_records, epoch_summaries, project_name = parse_log(log_path)

    train_records = [r for r in epoch_records if r["phase"] == "train"]
    val_records = [r for r in epoch_records if r["phase"] == "val"]

    last_record = val_records[-1] if val_records else (train_records[-1] if train_records else None)
    best_val = max(val_records, key=lambda x: float(x.get("mf1", x["epoch_mf1"])), default=None)
    last_summary = epoch_summaries[-1] if epoch_summaries else None

    max_epoch = None
    if val_records:
        max_epoch = int(val_records[-1]["max_epoch"])
    elif train_records:
        max_epoch = int(train_records[-1]["max_epoch"])
    elif last_summary and last_summary["max_epoch"] is not None:
        max_epoch = int(last_summary["max_epoch"])

    last_epoch = int(last_record["epoch"]) if last_record is not None else -1
    completed = bool(max_epoch is not None and last_epoch >= max_epoch)

    summary = {
        "project_dir": project_dir.name,
        "project_name": project_name,
        "case_name": infer_case_name(project_dir.name),
        "log_path": str(log_path),
        "completed": completed,
        "last_epoch": last_epoch,
        "max_epoch": max_epoch if max_epoch is not None else "",
        "last_train_mf1": float(train_records[-1]["mf1"]) if train_records and "mf1" in train_records[-1] else "",
        "last_val_mf1": float(val_records[-1]["mf1"]) if val_records and "mf1" in val_records[-1] else "",
        "last_val_miou": float(val_records[-1]["miou"]) if val_records and "miou" in val_records[-1] else "",
        "last_val_acc": float(val_records[-1]["acc"]) if val_records and "acc" in val_records[-1] else "",
        "best_val_mf1": float(best_val["mf1"]) if best_val and "mf1" in best_val else "",
        "best_val_miou": float(best_val["miou"]) if best_val and "miou" in best_val else "",
        "best_val_acc": float(best_val["acc"]) if best_val and "acc" in best_val else "",
        "best_epoch": int(best_val["epoch"]) if best_val else "",
        "epoch_time_min": float(last_summary["epoch_time_min"]) if last_summary else "",
        "remaining": last_summary["remaining"] if last_summary else "",
        "amp_enabled": last_summary["amp_enabled"] if last_summary else "",
        "amp_dtype": last_summary["amp_dtype"] if last_summary else "",
        "peak_allocated_mb": float(last_summary["peak_allocated_mb"]) if last_summary and last_summary["peak_allocated_mb"] is not None else "",
        "peak_reserved_mb": float(last_summary["peak_reserved_mb"]) if last_summary and last_summary["peak_reserved_mb"] is not None else "",
    }
    return summary, epoch_records, epoch_summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Ablation Log Summary",
        "",
        "| Case | Project Dir | Completed | Last Epoch | Best Epoch | Best Val mF1 | Best Val mIoU | Best Val Acc | Last Val mF1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['case_name']} | {row['project_dir']} | "
            f"{'yes' if row['completed'] else 'no'} | {row['last_epoch']} / {row['max_epoch']} | "
            f"{row['best_epoch']} | {row['best_val_mf1']} | {row['best_val_miou']} | "
            f"{row['best_val_acc']} | {row['last_val_mf1']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_logs(
    checkpoint_root: str | Path,
    output_dir: str | Path,
    projects: list[str] | None = None,
    only_completed: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    checkpoint_root = Path(checkpoint_root)
    output_dir = Path(output_dir)
    curve_dir = output_dir / "curves"

    if projects:
        project_dirs = [checkpoint_root / name for name in projects]
    else:
        project_dirs = sorted([p for p in checkpoint_root.iterdir() if p.is_dir()])

    summaries: list[dict[str, object]] = []
    completed_summaries: list[dict[str, object]] = []

    for project_dir in project_dirs:
        log_path = project_dir / "log.txt"
        if not log_path.exists():
            continue
        summary, epoch_records, epoch_summaries = summarize_project(project_dir)
        summaries.append(summary)
        if summary["completed"]:
            completed_summaries.append(summary)

        if only_completed and not summary["completed"]:
            continue

        write_csv(curve_dir / f"{project_dir.name}_epoch_metrics.csv", epoch_records)
        write_csv(curve_dir / f"{project_dir.name}_epoch_summary.csv", epoch_summaries)

    summaries.sort(key=lambda x: (not bool(x["completed"]), str(x["case_name"])))
    completed_summaries.sort(key=lambda x: str(x["case_name"]))

    write_csv(output_dir / "ablation_log_summary.csv", summaries)
    write_csv(output_dir / "ablation_log_summary_completed.csv", completed_summaries)
    write_markdown(output_dir / "ablation_log_summary.md", summaries)
    return summaries, completed_summaries, output_dir


def main() -> None:
    args = build_parser().parse_args()
    summaries, completed_summaries, _ = extract_logs(
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        projects=args.projects,
        only_completed=args.only_completed,
    )

    print(f"parsed_projects={len(summaries)}")
    print(f"completed_projects={len(completed_summaries)}")
    for row in completed_summaries:
        print(
            f"[completed] case={row['case_name']} "
            f"best_epoch={row['best_epoch']} best_val_mF1={row['best_val_mf1']} "
            f"best_val_mIoU={row['best_val_miou']} best_val_acc={row['best_val_acc']}"
        )


if __name__ == "__main__":
    main()
