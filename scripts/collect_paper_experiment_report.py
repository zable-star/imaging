from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "experiment",
    "num_runs",
    "mean_best_val_acc",
    "std_best_val_acc",
    "min_best_val_acc",
    "max_best_val_acc",
    "seeds",
    "aggregate_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect paper experiment aggregate CSVs into one report.")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--name-prefix", default="paper3090_")
    parser.add_argument("--output-csv", type=Path, default=Path("experiments/paper3090_combined_results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("writing/paper3090_training_report_2026-07-06.md"))
    return parser.parse_args()


def read_aggregate(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_rows(experiment_root: Path, name_prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for aggregate_path in sorted(experiment_root.glob(f"{name_prefix}*/aggregate_results.csv")):
        for row in read_aggregate(aggregate_path):
            rows.append(
                {
                    "experiment": row.get("experiment", aggregate_path.parent.name),
                    "num_runs": row.get("num_runs", ""),
                    "mean_best_val_acc": row.get("mean_best_val_acc", ""),
                    "std_best_val_acc": row.get("std_best_val_acc", ""),
                    "min_best_val_acc": row.get("min_best_val_acc", ""),
                    "max_best_val_acc": row.get("max_best_val_acc", ""),
                    "seeds": row.get("seeds", ""),
                    "aggregate_path": str(aggregate_path),
                }
            )
    return sorted(rows, key=lambda item: item["experiment"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value or "-"


def infer_group(experiment: str) -> str:
    if "smoke" in experiment:
        return "链路测试"
    if "military3class" in experiment:
        return "三类军事目标识别"
    if "single_gate" in experiment or "_gate1_" in experiment:
        return "单门/残余线索消融"
    if "_mean_" in experiment or "_attention_" in experiment or "_concat_" in experiment:
        return "融合方式对比"
    if "noise" in experiment or "dropout" in experiment or "atten" in experiment:
        return "鲁棒性验证"
    if "foreground" in experiment or "p99" in experiment:
        return "曝光匹配控制"
    if "truefalse" in experiment:
        return "真假目标主实验"
    return "其他"


def report_title(name_prefix: str) -> tuple[str, str]:
    if name_prefix.startswith("localgpu"):
        return (
            "本机 GPU 论文训练结果汇总",
            "本文件由 `scripts/collect_paper_experiment_report.py` 自动汇总 `localgpu_*` 实验生成。",
        )
    if name_prefix.startswith("paper3090"):
        return (
            "3090 论文训练结果汇总",
            "本文件由 `scripts/collect_paper_experiment_report.py` 自动汇总 `paper3090_*` 实验生成。",
        )
    return (
        f"{name_prefix} 训练结果汇总",
        f"本文件由 `scripts/collect_paper_experiment_report.py` 自动汇总 `{name_prefix}*` 实验生成。",
    )


def write_markdown(path: Path, rows: list[dict[str, str]], name_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title, source_line = report_title(name_prefix)
    lines = [
        f"# {title}",
        "",
        source_line,
        "主要用途是把多种子训练结果快速落到论文证据表中。",
        "",
    ]
    if name_prefix.startswith("localgpu"):
        lines.extend(
            [
                "注意：本机 GPU 结果用于日常推进、链路验证和趋势判断；短 epoch 结果不能替代 20-80 epoch 的论文主结果。",
                "",
            ]
        )

    if not rows:
        lines.extend(
            [
                "当前没有找到已完成的 `paper3090_*` aggregate 结果。",
                "",
                "建议先运行：",
                "",
                "```powershell",
                "powershell -ExecutionPolicy Bypass -File scripts\\run_3090_paper_experiments.ps1",
                "```",
                "",
            ]
        )
    else:
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(infer_group(row["experiment"]), []).append(row)

        for group, group_rows in groups.items():
            lines.extend(
                [
                    f"## {group}",
                    "",
                    "| experiment | runs | mean acc | std | min | max | seeds |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in group_rows:
                lines.append(
                    "| {experiment} | {runs} | {mean} | {std} | {minv} | {maxv} | {seeds} |".format(
                        experiment=row["experiment"],
                        runs=row["num_runs"],
                        mean=fmt_float(row["mean_best_val_acc"]),
                        std=fmt_float(row["std_best_val_acc"]),
                        minv=fmt_float(row["min_best_val_acc"]),
                        maxv=fmt_float(row["max_best_val_acc"]),
                        seeds=row["seeds"],
                    )
                )
            lines.append("")

        lines.extend(
            [
                "## 写论文时的使用口径",
                "",
                "- 三类军事目标识别用于说明小样本军事目标迁移是否比从零训练更稳定。",
                "- 真假目标主实验和单门消融用于回答激光选通是否提供了单帧图像之外的判别信息。",
                "- 如果单门长训练结果较高，应解释为当前仿真仍存在残余单帧形态线索；不要写成单门完全不可用。",
                "- 曝光匹配和鲁棒性实验用于排除亮度捷径，并说明方法在仿真退化下仍可工作。",
                "- 所有结果仍属于 Blender 仿真和 44 个精选军事模型条件下的验证，论文中不要写成真实外场系统已经验证。",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.experiment_root, args.name_prefix)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args.name_prefix)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")
    print(f"Collected experiments: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
