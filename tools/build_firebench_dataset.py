"""
Build FIRE-Bench train/val parquet files for verl-agent.

Usage:
    python tools/build_fire_bench_dataset.py \
        --fire_bench_root /home/xinle/FIRE-Bench \
        --output_dir $HOME/data/fire_bench \
        --val_ratio 0.2 \
        --seed 42
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd


def load_tasks(papers_dir: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(papers_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        instr_dir = task_dir / "instruction"

        gt_path = instr_dir / "instruction_gt.txt"
        user_path = instr_dir / "instruction.txt"

        if not gt_path.exists():
            print(f"  [skip] {task_dir.name} — no instruction_gt.txt")
            continue
        
        # Use user-facing instruction if available, else fall back to GT
        prompt_path = user_path if user_path.exists() else gt_path

        tasks.append(
            {
                "task_id": task_dir.name,
                "prompt": prompt_path.read_text().strip(),
                "instruction_gt": gt_path.read_text().strip(),
                "data_dir": str(task_dir / "data"),
            }
        )
    return tasks


def build_dummy_verl_row(task: dict) -> dict:
    """
    verl-agent data loader expects a 'data_source' column and a 'prompt'
    column that is a list of message dicts (chat format).
    """
    return {
        "data_source": "fire_bench",
        "task_id": task["task_id"],
        "prompt": [{"role": "user", "content": task["prompt"]}],
        "instruction_gt": task["instruction_gt"],
        "data_dir": task["data_dir"],
        # reward_model / extra_info left empty; reward is computed by the env
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"task_id": task["task_id"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fire_bench_root",
        default=os.environ.get("FIRE_BENCH_ROOT", "/home/xinle/FIRE-Bench"),
    )
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data/fire_bench"))
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    papers_dir = Path(args.fire_bench_root) / "benchmark" / "papers"
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers dir not found: {papers_dir}")

    print(f"Loading tasks from {papers_dir} ...")
    tasks = load_tasks(papers_dir)
    print(f"Found {len(tasks)} tasks")

    if not tasks:
        raise ValueError("No tasks found — check FIRE_BENCH_ROOT")

    random.seed(args.seed)
    random.shuffle(tasks)

    n_val = max(1, int(len(tasks) * args.val_ratio))
    val_tasks = tasks[:n_val]
    train_tasks = tasks[n_val:]
    print(f"Split: {len(train_tasks)} train / {len(val_tasks)} val")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [build_dummy_verl_row(t) for t in train_tasks]
    val_rows = [build_dummy_verl_row(t) for t in val_tasks]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    
    print(pd.DataFrame(train_rows).head())

    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")
    print("Done.")


if __name__ == "__main__":
    main()
