from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ablation.complexity import compare_with_full
from ablation.presets import (
    ABLATION_PRESETS,
    PAPER_DEFAULT_BOUNDARY_WEIGHT,
    PAPER_DEFAULT_LOSS_WEIGHTS,
    get_ablation_config,
)
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_ORDER = [
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
RESERVED_FORWARD_FLAGS = {
    "--ablation_case",
    "--project_name",
    "--checkpoint_root",
    "--vis_root",
    "--run_test_only",
}
PREFERRED_COLUMNS = [
    "case",
    "description",
    "status",
    "project_name",
    "elapsed_sec",
    "elapsed_min",
    "acc",
    "delta_acc_vs_full",
    "miou",
    "delta_miou_vs_full",
    "mf1",
    "delta_mf1_vs_full",
    "iou_1",
    "F1_1",
    "precision_1",
    "recall_1",
    "boundary_weight",
    "loss_weights_text",
    "params_m",
    "flops_g",
    "delta_params_m_vs_full",
    "delta_flops_g_vs_full",
    "params_reduction_pct_vs_full",
    "flops_reduction_pct_vs_full",
    "checkpoint_dir",
]
SUITE_TIMESTAMP_PREFIX = "ablation_suite_"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LiteCDNet ablation cases sequentially and aggregate results.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASE_ORDER,
        choices=DEFAULT_CASE_ORDER,
        help="Ablation cases to run in order.",
    )
    parser.add_argument(
        "--suite_name",
        type=str,
        default=f"ablation_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Suite name used for report folder and per-case project names.",
    )
    parser.add_argument(
        "--report_root",
        type=str,
        default="ablation_reports",
        help="Directory used to save aggregated suite reports.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints_ablation",
        help="Directory used by ablation checkpoints.",
    )
    parser.add_argument(
        "--vis_root",
        type=str,
        default="vis_ablation",
        help="Directory used by ablation visualizations.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a case when both best checkpoint and eval log already exist.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation for existing checkpoints.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the suite immediately when one case fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print and save the commands without running them.",
    )
    parser.add_argument(
        "--reuse_latest_unfinished",
        action="store_true",
        help="Reuse the latest unfinished suite automatically without prompting.",
    )
    parser.add_argument(
        "--no_reuse_prompt",
        action="store_true",
        help="Do not prompt for latest unfinished suite when --suite_name is omitted.",
    )
    return parser


def get_forward_arg_value(forward_args: list[str], flag: str, default: object = None) -> object:
    if flag not in forward_args:
        return default
    try:
        return forward_args[forward_args.index(flag) + 1]
    except Exception:
        return default


def validate_forward_args(forward_args: list[str]) -> None:
    conflict = [arg for arg in forward_args if arg in RESERVED_FORWARD_FLAGS]
    if conflict:
        joined = ", ".join(conflict)
        raise ValueError(
            f"These arguments are controlled by main_ablation_suite.py and should not be passed again: {joined}"
        )


def parse_metric_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    metric_line = ""
    for line in lines:
        if "acc=" in line and "miou=" in line and "mf1=" in line:
            metric_line = line.strip()

    if not metric_line:
        return {}

    metrics: dict[str, float] = {}
    for token in metric_line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def inspect_case_checkpoint(checkpoint_dir: Path, max_epochs: int | None) -> dict[str, object]:
    info: dict[str, object] = {
        "mode": "scratch",
        "epoch_to_start": 0,
        "remaining_epochs": None,
        "best_exists": (checkpoint_dir / "best_ckpt.pt").exists(),
        "last_exists": (checkpoint_dir / "last_ckpt.pt").exists(),
    }
    last_ckpt = checkpoint_dir / "last_ckpt.pt"
    if not last_ckpt.exists():
        return info

    try:
        checkpoint = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        epoch_to_start = int(checkpoint.get("epoch_id", -1)) + 1
        info["mode"] = "resume"
        info["epoch_to_start"] = max(epoch_to_start, 0)
        if max_epochs is not None:
            info["remaining_epochs"] = max(0, max_epochs - max(epoch_to_start, 0))
            if info["remaining_epochs"] == 0:
                info["mode"] = "completed"
    except Exception as exc:
        info["mode"] = f"resume_detect_failed:{exc.__class__.__name__}"
    return info


def build_suite_plan_rows(suite_args: argparse.Namespace, forward_args: list[str]) -> list[dict[str, object]]:
    try:
        max_epochs = int(get_forward_arg_value(forward_args, "--max_epochs", 150))
    except Exception:
        max_epochs = 150

    rows: list[dict[str, object]] = []
    for case_name in suite_args.cases:
        project_name = f"{suite_args.suite_name}_{case_name}"
        checkpoint_dir = (PROJECT_ROOT / suite_args.checkpoint_root / project_name).resolve()
        info = inspect_case_checkpoint(checkpoint_dir, max_epochs)
        rows.append(
            {
                "case": case_name,
                "project_name": project_name,
                "mode": "eval_only" if suite_args.eval_only else info.get("mode", "scratch"),
                "status_cn": {
                    "scratch": "从头开始",
                    "resume": "继续训练",
                    "completed": "已完成",
                    "eval_only": "仅评估",
                }.get("eval_only" if suite_args.eval_only else info.get("mode", "scratch"), "状态异常"),
                "epoch_to_start": int(info.get("epoch_to_start", 0) or 0),
                "remaining_epochs": max_epochs if info.get("remaining_epochs") is None else info.get("remaining_epochs"),
                "best_exists": "Y" if info.get("best_exists") else "N",
                "last_exists": "Y" if info.get("last_exists") else "N",
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
    return rows


def summarize_plan_rows(plan_rows: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "scratch": 0,
        "resume": 0,
        "completed": 0,
        "eval_only": 0,
        "other": 0,
    }
    for row in plan_rows:
        mode = str(row.get("mode", "other"))
        if mode in summary:
            summary[mode] += 1
        else:
            summary["other"] += 1
    return summary


def _extract_suite_name_from_case_dir(case_dir_name: str) -> str | None:
    for case_name in DEFAULT_CASE_ORDER:
        suffix = f"_{case_name}"
        if case_dir_name.endswith(suffix):
            return case_dir_name[: -len(suffix)]
    return None


def discover_candidate_suite_names(checkpoint_root: Path, report_root: Path) -> list[str]:
    names: set[str] = set()

    if checkpoint_root.exists():
        for child in checkpoint_root.iterdir():
            if not child.is_dir():
                continue
            suite_name = _extract_suite_name_from_case_dir(child.name)
            if suite_name:
                names.add(suite_name)

    if report_root.exists():
        for child in report_root.iterdir():
            if child.is_dir():
                names.add(child.name)

    pattern = re.compile(rf"^{re.escape(SUITE_TIMESTAMP_PREFIX)}\d{{8}}_\d{{6}}$")
    filtered = [name for name in names if pattern.match(name)]
    filtered.sort(reverse=True)
    return filtered


def find_latest_unfinished_suite(
    current_suite_name: str,
    checkpoint_root: Path,
    report_root: Path,
    forward_args: list[str],
    eval_only: bool,
) -> tuple[str | None, list[dict[str, object]]]:
    for suite_name in discover_candidate_suite_names(checkpoint_root, report_root):
        if suite_name == current_suite_name:
            continue

        temp_args = argparse.Namespace(
            suite_name=suite_name,
            checkpoint_root=str(checkpoint_root.relative_to(PROJECT_ROOT)) if checkpoint_root.is_relative_to(PROJECT_ROOT) else str(checkpoint_root),
            eval_only=eval_only,
            cases=DEFAULT_CASE_ORDER,
        )
        plan_rows = build_suite_plan_rows(temp_args, forward_args)
        summary = summarize_plan_rows(plan_rows)
        has_progress = any(row.get("last_exists") == "Y" or row.get("best_exists") == "Y" for row in plan_rows)
        unfinished = summary["resume"] > 0 or (
            has_progress and (summary["scratch"] > 0 or summary["completed"] < len(plan_rows))
        )
        if unfinished:
            return suite_name, plan_rows
    return None, []


def maybe_reuse_latest_unfinished_suite(
    suite_args: argparse.Namespace,
    forward_args: list[str],
    raw_argv: list[str],
) -> argparse.Namespace:
    if "--suite_name" in raw_argv:
        return suite_args
    if suite_args.no_reuse_prompt:
        return suite_args

    checkpoint_root = (PROJECT_ROOT / suite_args.checkpoint_root).resolve()
    report_root = (PROJECT_ROOT / suite_args.report_root).resolve()
    latest_suite_name, latest_plan_rows = find_latest_unfinished_suite(
        current_suite_name=suite_args.suite_name,
        checkpoint_root=checkpoint_root,
        report_root=report_root,
        forward_args=forward_args,
        eval_only=suite_args.eval_only,
    )
    if latest_suite_name is None:
        return suite_args

    print("=" * 96)
    print("Detected latest unfinished suite candidate.")
    print(f"Current auto-generated suite_name: {suite_args.suite_name}")
    print(f"Latest unfinished suite_name: {latest_suite_name}")
    print(render_suite_plan_table(latest_plan_rows, argparse.Namespace(suite_name=latest_suite_name, cases=DEFAULT_CASE_ORDER), forward_args))
    print("")

    reuse = False
    if suite_args.reuse_latest_unfinished:
        reuse = True
        print("Auto action: reuse latest unfinished suite because --reuse_latest_unfinished is enabled.")
    else:
        prompt = f"Reuse latest unfinished suite '{latest_suite_name}' instead of creating a new timestamped suite? [Y/n]: "
        try:
            answer = input(prompt).strip().lower()
            reuse = answer in {"", "y", "yes"}
        except EOFError:
            print("No interactive input available; keep current auto-generated suite_name.")
            reuse = False
        except KeyboardInterrupt:
            print("\nPrompt cancelled; keep current auto-generated suite_name.")
            reuse = False

    if reuse:
        suite_args.suite_name = latest_suite_name
        print(f"Using reused suite_name: {suite_args.suite_name}")
    else:
        print(f"Using new suite_name: {suite_args.suite_name}")
    print("=" * 96)
    return suite_args


def build_suite_plan_rows(suite_args: argparse.Namespace, forward_args: list[str]) -> list[dict[str, object]]:
    try:
        max_epochs = int(get_forward_arg_value(forward_args, "--max_epochs", 150))
    except Exception:
        max_epochs = 150
    try:
        img_size = int(get_forward_arg_value(forward_args, "--img_size", 256))
    except Exception:
        img_size = 256
    try:
        n_class = int(get_forward_arg_value(forward_args, "--n_class", 2))
    except Exception:
        n_class = 2

    rows: list[dict[str, object]] = []
    for case_name in suite_args.cases:
        project_name = f"{suite_args.suite_name}_{case_name}"
        checkpoint_dir = (PROJECT_ROOT / suite_args.checkpoint_root / project_name).resolve()
        info = inspect_case_checkpoint(checkpoint_dir, max_epochs)
        preset = get_ablation_config(case_name)
        complexity = compare_with_full(case_name, img_size=img_size, n_class=n_class)
        mode = "eval_only" if suite_args.eval_only else info.get("mode", "scratch")
        rows.append(
            {
                "case": case_name,
                "project_name": project_name,
                "mode": mode,
                "status_cn": {
                    "scratch": "从头开始",
                    "resume": "继续训练",
                    "completed": "已完成",
                    "eval_only": "仅评估",
                }.get(mode, "状态异常"),
                "epoch_to_start": int(info.get("epoch_to_start", 0) or 0),
                "remaining_epochs": max_epochs if info.get("remaining_epochs") is None else info.get("remaining_epochs"),
                "best_exists": "Y" if info.get("best_exists") else "N",
                "last_exists": "Y" if info.get("last_exists") else "N",
                "boundary_weight": preset["boundary_weight"],
                "loss_weights_text": "/".join(f"{float(x):.1f}" for x in preset["loss_weights"]),
                "params_m": complexity["params_m"],
                "flops_g": complexity["flops_g"],
                "delta_params_m_vs_full": complexity["delta_params_m_vs_full"],
                "delta_flops_g_vs_full": complexity["delta_flops_g_vs_full"],
                "params_reduction_pct_vs_full": complexity["params_reduction_pct_vs_full"],
                "flops_reduction_pct_vs_full": complexity["flops_reduction_pct_vs_full"],
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
    return rows


def render_suite_plan_table(plan_rows: list[dict[str, object]], suite_args: argparse.Namespace, forward_args: list[str]) -> str:
    lines = [
        f"===== Suite Pre-Scan: {suite_args.suite_name} =====",
        f"cases={', '.join(suite_args.cases)}",
        f"forwarded_args={' '.join(forward_args) if forward_args else '(none)'}",
        f"paper_defaults: L_boundary={PAPER_DEFAULT_BOUNDARY_WEIGHT} | loss_weights={'/'.join(map(str, PAPER_DEFAULT_LOSS_WEIGHTS))}",
        "",
        f"{'case':<22}{'状态':<12}{'start':<8}{'remain':<8}{'L_b':<6}{'msw':<16}{'Params(M)':<11}{'FLOPs(G)':<10}{'dP_vs_full':<12}{'dF_vs_full':<12}",
        "-" * 130,
    ]
    for row in plan_rows:
        remaining_text = str(row["remaining_epochs"]) if row["remaining_epochs"] is not None else "-"
        lines.append(
            f"{str(row['case']):<22}"
            f"{str(row['status_cn']):<12}"
            f"{str(row['epoch_to_start']):<8}"
            f"{remaining_text:<8}"
            f"{float(row['boundary_weight']):<6.1f}"
            f"{str(row['loss_weights_text']):<16}"
            f"{float(row['params_m']):<11.2f}"
            f"{float(row['flops_g']):<10.2f}"
            f"{float(row['delta_params_m_vs_full']):<+12.2f}"
            f"{float(row['delta_flops_g_vs_full']):<+12.2f}"
        )
        lines.append(f"  project={row['project_name']} | ckpt_last={row['last_exists']} | ckpt_best={row['best_exists']}")
    return "\n".join(lines)


def write_suite_plan(plan_rows: list[dict[str, object]], output_path: Path) -> None:
    columns = [
        "case",
        "mode",
        "status_cn",
        "epoch_to_start",
        "remaining_epochs",
        "boundary_weight",
        "loss_weights_text",
        "params_m",
        "flops_g",
        "delta_params_m_vs_full",
        "delta_flops_g_vs_full",
        "params_reduction_pct_vs_full",
        "flops_reduction_pct_vs_full",
        "last_exists",
        "best_exists",
        "project_name",
        "checkpoint_dir",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in plan_rows:
            writer.writerow(row)


def render_suite_plan_table(plan_rows: list[dict[str, object]], suite_args: argparse.Namespace, forward_args: list[str]) -> str:
    header = [
        f"===== Suite Pre-Scan: {suite_args.suite_name} =====",
        f"cases={', '.join(suite_args.cases)}",
        f"forwarded_args={' '.join(forward_args) if forward_args else '(none)'}",
        f"paper_defaults: L_boundary={PAPER_DEFAULT_BOUNDARY_WEIGHT} | loss_weights={'/'.join(map(str, PAPER_DEFAULT_LOSS_WEIGHTS))}",
        "",
        f"{'case':<22}{'状态':<12}{'start':<8}{'remain':<8}{'L_b':<6}{'msw':<16}{'Params(M)':<11}{'FLOPs(G)':<10}{'dP_vs_full':<12}{'dF_vs_full':<12}",
        "-" * 130,
    ]
    lines = header[:]
    for row in plan_rows:
        remaining_text = str(row["remaining_epochs"]) if row["remaining_epochs"] is not None else "-"
        lines.append(
            f"{str(row['case']):<22}"
            f"{str(row['status_cn']):<12}"
            f"{str(row['epoch_to_start']):<8}"
            f"{remaining_text:<8}"
            f"{float(row['boundary_weight']):<6.1f}"
            f"{str(row['loss_weights_text']):<16}"
            f"{float(row['params_m']):<11.2f}"
            f"{float(row['flops_g']):<10.2f}"
            f"{float(row['delta_params_m_vs_full']):<+12.2f}"
            f"{float(row['delta_flops_g_vs_full']):<+12.2f}"
        )
        lines.append(f"  project={row['project_name']} | ckpt_last={row['last_exists']} | ckpt_best={row['best_exists']}")
    return "\n".join(lines)


def write_suite_plan(plan_rows: list[dict[str, object]], output_path: Path) -> None:
    columns = [
        "case",
        "mode",
        "status_cn",
        "epoch_to_start",
        "remaining_epochs",
        "boundary_weight",
        "loss_weights_text",
        "params_m",
        "flops_g",
        "delta_params_m_vs_full",
        "delta_flops_g_vs_full",
        "params_reduction_pct_vs_full",
        "flops_reduction_pct_vs_full",
        "last_exists",
        "best_exists",
        "project_name",
        "checkpoint_dir",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in plan_rows:
            writer.writerow(row)


def run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def build_case_command(
    case_name: str,
    project_name: str,
    suite_args: argparse.Namespace,
    forward_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-X",
        "utf8",
        "main_ablation.py",
        "--ablation_case",
        case_name,
        "--project_name",
        project_name,
        "--checkpoint_root",
        suite_args.checkpoint_root,
        "--vis_root",
        suite_args.vis_root,
    ]
    if suite_args.eval_only:
        command.extend(["--run_test_only", "True"])
    command.extend(forward_args)
    return command


def make_case_row(
    case_name: str,
    suite_args: argparse.Namespace,
    metrics: dict[str, float] | None = None,
    status: str = "pending",
    elapsed_sec: float = 0.0,
) -> dict[str, object]:
    project_name = f"{suite_args.suite_name}_{case_name}"
    checkpoint_dir = str((PROJECT_ROOT / suite_args.checkpoint_root / project_name).resolve())
    preset = get_ablation_config(case_name)
    complexity = compare_with_full(case_name)
    row: dict[str, object] = {
        "case": case_name,
        "description": ABLATION_PRESETS[case_name]["description"],
        "status": status,
        "project_name": project_name,
        "elapsed_sec": round(elapsed_sec, 2),
        "elapsed_min": round(elapsed_sec / 60.0, 3),
        "boundary_weight": preset["boundary_weight"],
        "loss_weights_text": "/".join(f"{float(x):.1f}" for x in preset["loss_weights"]),
        "params_m": round(complexity["params_m"], 4),
        "flops_g": round(complexity["flops_g"], 4),
        "delta_params_m_vs_full": round(complexity["delta_params_m_vs_full"], 4),
        "delta_flops_g_vs_full": round(complexity["delta_flops_g_vs_full"], 4),
        "params_reduction_pct_vs_full": round(complexity["params_reduction_pct_vs_full"], 4),
        "flops_reduction_pct_vs_full": round(complexity["flops_reduction_pct_vs_full"], 4),
        "checkpoint_dir": checkpoint_dir,
    }
    if metrics:
        row.update(metrics)
    return row


def add_baseline_deltas(rows: list[dict[str, object]]) -> None:
    baseline = next((row for row in rows if row.get("case") == "full"), None)
    if baseline is None:
        return

    for metric_name, delta_name in (
        ("acc", "delta_acc_vs_full"),
        ("miou", "delta_miou_vs_full"),
        ("mf1", "delta_mf1_vs_full"),
    ):
        baseline_value = baseline.get(metric_name)
        if not isinstance(baseline_value, (int, float)):
            continue
        for row in rows:
            value = row.get(metric_name)
            if isinstance(value, (int, float)):
                row[delta_name] = round(float(value) - float(baseline_value), 6)


def collect_all_columns(rows: list[dict[str, object]]) -> list[str]:
    extra_columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in PREFERRED_COLUMNS and key not in extra_columns:
                extra_columns.append(key)
    return PREFERRED_COLUMNS + extra_columns


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    columns = collect_all_columns(rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(
    rows: list[dict[str, object]],
    output_path: Path,
    suite_args: argparse.Namespace,
    forward_args: list[str],
) -> None:
    columns = [
        "case",
        "status",
        "miou",
        "delta_miou_vs_full",
        "mf1",
        "delta_mf1_vs_full",
        "acc",
        "elapsed_min",
        "description",
    ]
    lines = [
        f"# Ablation Suite Summary: {suite_args.suite_name}",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Cases: {', '.join(suite_args.cases)}",
        f"- Forwarded args: {' '.join(forward_args) if forward_args else '(none)'}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(col, "")) for col in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands(commands: list[dict[str, str]], output_path: Path) -> None:
    lines = []
    for item in commands:
        lines.append(f"[{item['case']}]")
        lines.append(item["command"])
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(suite_args: argparse.Namespace, forward_args: list[str]) -> list[dict[str, object]]:
    report_dir = (PROJECT_ROOT / suite_args.report_root / suite_args.suite_name).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    command_records: list[dict[str, str]] = []
    suite_started = time.time()
    plan_rows = build_suite_plan_rows(suite_args, forward_args)
    plan_text = render_suite_plan_table(plan_rows, suite_args, forward_args)
    print(plan_text)
    print("")
    (report_dir / "suite_plan_preview.txt").write_text(plan_text + "\n", encoding="utf-8")
    write_suite_plan(plan_rows, report_dir / "suite_plan_preview.csv")

    for index, case_name in enumerate(suite_args.cases, start=1):
        project_name = f"{suite_args.suite_name}_{case_name}"
        checkpoint_dir = (PROJECT_ROOT / suite_args.checkpoint_root / project_name).resolve()
        best_ckpt = checkpoint_dir / "best_ckpt.pt"
        eval_log = checkpoint_dir / "log_eval.txt"
        stdout_log = report_dir / f"{index:02d}_{case_name}.stdout.log"
        max_epochs = None
        max_epochs_value = get_forward_arg_value(forward_args, "--max_epochs", None)
        if max_epochs_value is not None:
            try:
                max_epochs = int(max_epochs_value)
            except Exception:
                max_epochs = None
        checkpoint_info = inspect_case_checkpoint(checkpoint_dir, max_epochs)

        command = build_case_command(case_name, project_name, suite_args, forward_args)
        command_records.append({"case": case_name, "command": " ".join(command)})
        print(f"\n===== [{index}/{len(suite_args.cases)}] {case_name} =====")
        print(f"Description: {ABLATION_PRESETS[case_name]['description']}")
        print(f"Project name: {project_name}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Run mode: {'eval_only' if suite_args.eval_only else checkpoint_info['mode']}")
        if checkpoint_info.get("mode") == "resume":
            print(
                f"Resume detail: epoch_to_start={checkpoint_info['epoch_to_start']}"
                + (
                    f", remaining_epochs={checkpoint_info['remaining_epochs']}"
                    if checkpoint_info.get("remaining_epochs") is not None
                    else ""
                )
            )
        case_complexity = compare_with_full(case_name)
        case_preset = get_ablation_config(case_name)
        print(
            f"Loss config: L_boundary={case_preset['boundary_weight']} | "
            f"loss_weights={case_preset['loss_weights']}"
        )
        print(
            f"Complexity vs full: Params={case_complexity['params_m']:.2f}M "
            f"(Δ {case_complexity['delta_params_m_vs_full']:+.2f}M, {case_complexity['params_reduction_pct_vs_full']:+.2f}%) | "
            f"FLOPs={case_complexity['flops_g']:.2f}G "
            f"(Δ {case_complexity['delta_flops_g_vs_full']:+.2f}G, {case_complexity['flops_reduction_pct_vs_full']:+.2f}%)"
        )
        print(f"best_ckpt_exists={best_ckpt.exists()} | log_eval_exists={eval_log.exists()}")
        print(" ".join(command))

        if suite_args.dry_run:
            rows.append(make_case_row(case_name, suite_args, status="dry_run"))
            continue

        if suite_args.skip_existing and best_ckpt.exists() and eval_log.exists() and not suite_args.eval_only:
            print("Action: skip existing completed case.")
            metrics = parse_metric_log(eval_log)
            rows.append(make_case_row(case_name, suite_args, metrics, status="skipped_existing"))
            continue

        started_at = time.time()
        return_code = run_command(command, stdout_log)
        elapsed_sec = time.time() - started_at

        if return_code != 0:
            row = make_case_row(case_name, suite_args, status=f"failed({return_code})", elapsed_sec=elapsed_sec)
            rows.append(row)
            suite_elapsed_hours = (time.time() - suite_started) / 3600.0
            print(
                f"Case finished with failure. elapsed={elapsed_sec / 60.0:.2f} min | "
                f"suite_elapsed={suite_elapsed_hours:.2f} h"
            )
            if suite_args.stop_on_error:
                break
            continue

        metrics = parse_metric_log(eval_log)
        status = "ok" if metrics else "ok_no_metrics"
        rows.append(make_case_row(case_name, suite_args, metrics, status=status, elapsed_sec=elapsed_sec))
        suite_elapsed_sec = time.time() - suite_started
        completed_cases = len(rows)
        avg_case_sec = suite_elapsed_sec / max(completed_cases, 1)
        remaining_case_count = len(suite_args.cases) - index
        rough_remaining_sec = avg_case_sec * remaining_case_count
        print(
            f"Case finished. elapsed={elapsed_sec / 60.0:.2f} min | "
            f"suite_elapsed={suite_elapsed_sec / 3600.0:.2f} h | "
            f"rough_remaining={rough_remaining_sec / 3600.0:.2f} h"
        )

    add_baseline_deltas(rows)
    if suite_args.dry_run:
        write_csv(rows, report_dir / "summary_dry_run.csv")
        write_json(rows, report_dir / "summary_dry_run.json")
        write_markdown(rows, report_dir / "summary_dry_run.md", suite_args, forward_args)
    else:
        write_csv(rows, report_dir / "summary.csv")
        write_json(rows, report_dir / "summary.json")
        write_markdown(rows, report_dir / "summary.md", suite_args, forward_args)
    write_commands(command_records, report_dir / "commands.txt")

    suite_config = {
        "suite_name": suite_args.suite_name,
        "cases": suite_args.cases,
        "checkpoint_root": suite_args.checkpoint_root,
        "vis_root": suite_args.vis_root,
        "report_root": suite_args.report_root,
        "skip_existing": suite_args.skip_existing,
        "eval_only": suite_args.eval_only,
        "dry_run": suite_args.dry_run,
        "forward_args": forward_args,
    }
    (report_dir / "suite_config.json").write_text(
        json.dumps(suite_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def print_brief_summary(rows: list[dict[str, object]]) -> None:
    print("\n===== Suite Summary =====")
    for row in rows:
        parts = [
            f"case={row.get('case')}",
            f"status={row.get('status')}",
        ]
        if "miou" in row:
            parts.append(f"miou={float(row['miou']):.5f}")
        if "mf1" in row:
            parts.append(f"mf1={float(row['mf1']):.5f}")
        if "delta_miou_vs_full" in row:
            parts.append(f"delta_miou_vs_full={float(row['delta_miou_vs_full']):+.5f}")
        print(" | ".join(parts))


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    suite_args, forward_args = parser.parse_known_args()
    validate_forward_args(forward_args)
    suite_args = maybe_reuse_latest_unfinished_suite(suite_args, forward_args, raw_argv)
    rows = run_suite(suite_args, forward_args)
    print_brief_summary(rows)


if __name__ == "__main__":
    main()
