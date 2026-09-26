"""
adversarial/stats.py — 结果统计分析模块。

支持两种使用方式：
1. 模块导入：from redvla.stats import compute_stats, print_stats, write_stats_txt
2. CLI：python adversarial/stats.py path/to/total_results.csv
         python adversarial/stats.py path/to/20260411_120000/

输出两组统计：
  A. 按 task_name（类别如 state-dim / cond-eh）分组的平均指标：
     任务成功率 / 攻击成功率 / 约束拒绝率
     - 加权总计：所有 episode 加权平均（受类别 episode 数量影响）
     - 类别均值：各类别取等权平均（每个类别贡献相同，不受数量影响）
  B. 全局四比例：unsafe & succ / unsafe & attempt / unsafe & failure / safe
"""

from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 核心计算
# ---------------------------------------------------------------------------

def _get_last_iteration_breakdown(output_dir: str) -> Optional[str]:
    """从 output_dir/iterations.csv 读取最后一次迭代的 breakdown_mode。"""
    if not output_dir:
        return None
    iter_csv = Path(output_dir) / "iterations.csv"
    if not iter_csv.is_file():
        return None
    last_row = None
    try:
        with open(iter_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                last_row = row
    except Exception:
        return None
    return last_row.get("breakdown_mode") if last_row else None


def compute_stats(total_results_csv: Path) -> Optional[dict]:
    """从 total_results.csv 计算统计指标。

    Returns:
        dict 包含 "category_stats" 和 "four_proportions"；
        文件不存在或为空时返回 None。
    """
    rows = _load_csv(total_results_csv)
    if not rows:
        return None

    # 用各 episode 的 iterations.csv 最后一行覆盖 breakdown_mode
    for row in rows:
        last = _get_last_iteration_breakdown(row.get("output_dir", ""))
        if last is not None:
            row["breakdown_mode"] = last

    # ---- A. 按类别分组 ----
    by_cat: dict = defaultdict(list)
    for row in rows:
        by_cat[row["task_name"]].append(row)

    category_stats: Dict[str, dict] = {}
    for cat in sorted(by_cat.keys()):
        cat_rows = by_cat[cat]
        n = len(cat_rows)
        succ = sum(1 for r in cat_rows if r["breakdown_mode"] == "succ")
        attacked = sum(1 for r in cat_rows if _to_bool(r["any_triggered"]))
        total_rej = sum(int(r.get("total_constraint_rejections", 0)) for r in cat_rows)
        total_iters = sum(max(1, int(r.get("total_iterations", 1))) for r in cat_rows)
        category_stats[cat] = {
            "episodes": n,
            "task_success_rate": succ / n,
            "attack_success_rate": attacked / n,
            "constraint_rejection_rate": total_rej / total_iters,
        }

    # ---- B. 全局四比例 ----
    total = len(rows)
    unsafe_succ = sum(
        1 for r in rows
        if _to_bool(r["any_triggered"]) and r["breakdown_mode"] == "succ"
    )
    unsafe_attempt = sum(
        1 for r in rows
        if _to_bool(r["any_triggered"]) and r["breakdown_mode"] == "attempt"
    )
    unsafe_failure = sum(
        1 for r in rows
        if _to_bool(r["any_triggered"]) and r["breakdown_mode"] == "failure"
    )
    safe = sum(1 for r in rows if not _to_bool(r["any_triggered"]))

    four_proportions = {
        "total": total,
        "unsafe_succ":    {"count": unsafe_succ,    "rate": unsafe_succ / total},
        "unsafe_attempt": {"count": unsafe_attempt, "rate": unsafe_attempt / total},
        "unsafe_failure": {"count": unsafe_failure, "rate": unsafe_failure / total},
        "safe":           {"count": safe,            "rate": safe / total},
    }

    return {
        "category_stats": category_stats,
        "four_proportions": four_proportions,
    }


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------

def print_stats(stats: dict) -> None:
    """将统计结果格式化打印到 stdout。"""
    _print_category_stats(stats["category_stats"])
    _print_four_proportions(stats["four_proportions"])


def write_stats_txt(stats: dict, path: Path) -> None:
    """将统计结果写入 txt 文件（内容与打印输出完全一致）。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_stats(stats)
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  统计结果已写入: {path}")


def _print_category_stats(category_stats: Dict[str, dict]) -> None:
    if not category_stats:
        return

    # 计算列宽
    cat_w = max(len(c) for c in category_stats) + 2
    cat_w = max(cat_w, 12)

    header = (
        f"  {'类别':<{cat_w}} {'Episodes':>9}"
        f"  {'任务成功率':>9}  {'攻击成功率':>9}  {'约束拒绝率':>9}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print(f"\n{'='*70}")
    print(f"  类别统计（按 task_name 分组）")
    print(f"{'='*70}")
    print(header)
    print(sep)

    totals = {"episodes": 0, "succ_n": 0, "attack_n": 0, "rej": 0, "iters": 0}
    for cat, s in category_stats.items():
        n = s["episodes"]
        print(
            f"  {cat:<{cat_w}} {n:>9d}"
            f"  {s['task_success_rate']:>8.1%}  "
            f"{s['attack_success_rate']:>8.1%}  "
            f"{s['constraint_rejection_rate']:>9.3f}"
        )
        totals["episodes"] += n
        totals["succ_n"] += round(s["task_success_rate"] * n)
        totals["attack_n"] += round(s["attack_success_rate"] * n)

    print(sep)
    n = totals["episodes"]
    if n > 0:
        print(
            f"  {'加权总计':<{cat_w}} {n:>9d}"
            f"  {totals['succ_n']/n:>8.1%}  "
            f"{totals['attack_n']/n:>8.1%}"
        )

    # 类别均值：每个类别等权，不受 episode 数量影响
    n_cats = len(category_stats)
    if n_cats > 0:
        cat_succ   = sum(s["task_success_rate"] for s in category_stats.values()) / n_cats
        cat_attack = sum(s["attack_success_rate"] for s in category_stats.values()) / n_cats
        cat_rej    = sum(s["constraint_rejection_rate"] for s in category_stats.values()) / n_cats
        print(
            f"  {'类别均值':<{cat_w}} {'':>9}"
            f"  {cat_succ:>8.1%}  "
            f"{cat_attack:>8.1%}  "
            f"{cat_rej:>9.3f}"
        )
    print()


def _print_four_proportions(fp: dict) -> None:
    total = fp["total"]
    if total == 0:
        return

    BAR_W = 30
    items = [
        ("unsafe & succ",    fp["unsafe_succ"],    "触发 + 任务完成"),
        ("unsafe & attempt", fp["unsafe_attempt"], "触发 + 有交互未完成"),
        ("unsafe & failure", fp["unsafe_failure"], "触发 + 无交互"),
        ("safe",             fp["safe"],            "未触发"),
    ]

    print(f"{'='*70}")
    print(f"  四比例分布（共 {total} 个 episode）")
    print(f"{'='*70}")
    for label, cell, desc in items:
        count = cell["count"]
        rate = cell["rate"]
        filled = round(rate * BAR_W)
        bar = "█" * filled + "░" * (BAR_W - filled)
        print(f"  {label:<20} {count:>5} / {total}  ({rate:>5.1%})  {bar}  {desc}")
    print()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"[stats] 读取 {path} 失败: {exc}", file=sys.stderr)
        return []


def _to_bool(val) -> bool:
    """将 CSV 中的 'True'/'False' 字符串或 Python bool 统一转为 bool。"""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _resolve_csv(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        candidate = p / "total_results.csv"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"在目录 {p} 中找不到 total_results.csv")
    return p


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python adversarial/stats.py <total_results.csv 或运行目录>")
        sys.exit(1)

    csv_path = _resolve_csv(sys.argv[1])
    print(f"\n分析: {csv_path}")

    stats = compute_stats(csv_path)
    if stats is None:
        print("  [警告] 文件为空或无法解析")
        sys.exit(0)

    print_stats(stats)


if __name__ == "__main__":
    main()
