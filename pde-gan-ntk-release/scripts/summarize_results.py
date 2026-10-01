#!/usr/bin/env python3
"""Create portable CSV and Markdown summaries from reproduced run artifacts."""

import csv
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "denn" / "config"
RUNS_DIR = ROOT / "experiments" / "runs"
RESULTS_DIR = ROOT / "results"
PROBLEMS = ("pos", "lap1", "lap2", "wav", "hea", "burv")
METHODS = ("gan", "lsgan", "wgan")


def load_series(path):
    if not path.exists():
        return np.asarray([])
    return np.asarray(np.load(path, allow_pickle=False)).reshape(-1)


def number(value):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.12e}"


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect(pkey, trainer):
    with (CONFIG_DIR / f"{pkey}.yaml").open() as handle:
        config = yaml.safe_load(handle)
    expected = int(config["training"]["niters"])
    run_dir = RUNS_DIR / config["training"]["dirname"]

    val_name = "mse_val.npy" if trainer == "ntk" else "val_mse.npy"
    train_name = "mse_train.npy" if trainer == "ntk" else "train_mse.npy"
    val = load_series(run_dir / val_name)
    train = load_series(run_dir / train_name)
    checkpoint_required = trainer == "ntk"
    complete = len(val) == expected and (run_dir / "config.yaml").exists()
    if checkpoint_required:
        complete = complete and (run_dir / "generator_final.pt").exists()

    return {
        "pkey": pkey,
        "trainer": trainer,
        "status": "complete" if complete else "incomplete",
        "expected_epochs": expected,
        "recorded_epochs": len(val),
        "final_train_mse": number(train[-1] if len(train) else None),
        "best_train_mse": number(train.min() if len(train) else None),
        "final_val_mse": number(val[-1] if len(val) else None),
        "best_val_mse": number(val.min() if len(val) else None),
        "result_dir": display_path(run_dir),
    }


def write_summary(name, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    md_path = RESULTS_DIR / f"{name}.md"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as handle:
        handle.write(
            "| Config | Status | Epochs | Final train MSE | "
            "Final validation MSE | Best validation MSE |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['pkey']} | {row['status']} | "
                f"{row['recorded_epochs']}/{row['expected_epochs']} | "
                f"{row['final_train_mse']} | {row['final_val_mse']} | "
                f"{row['best_val_mse']} |\n"
            )
    print(f"Wrote {display_path(csv_path)}")
    print(f"Wrote {display_path(md_path)}")


def main():
    deqgan_rows = [collect(problem, "deqgan") for problem in PROBLEMS]
    ntk_rows = [
        collect(f"{problem}_ntk_{method}", "ntk")
        for problem in PROBLEMS
        for method in METHODS
    ]
    write_summary("reproduced_deqgan_results", deqgan_rows)
    write_summary("reproduced_ntk_results", ntk_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
