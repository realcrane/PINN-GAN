#!/usr/bin/env python3
"""Audit every README experiment and write portable CSV/Markdown summaries."""

import argparse
import csv
import platform
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "denn" / "config"
RUNS_DIR = ROOT / "experiments" / "runs"
RESULTS_DIR = ROOT / "results"
PROBLEMS = ("pos", "lap1", "lap2", "kle", "rea", "burv")
METHODS = ("gan", "lsgan", "wgan")
DISPLAY = {
    "pos": "Poisson",
    "lap1": "Laplace 1",
    "lap2": "Laplace 2",
    "kle": "Klein-Gordon",
    "rea": "Reaction-diffusion",
    "burv": "Viscous Burgers",
}
RUN_STEMS = {
    "pos": "Pos",
    "lap1": "Lap1",
    "lap2": "Lap2",
    "kle": "Kle",
    "rea": "Rea",
    "burv": "Burv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status-dir",
        type=Path,
        help="directory containing status_cuda_*.csv (defaults to newest full run)",
    )
    parser.add_argument("--name", default="readme_validation_results")
    return parser.parse_args()


def latest_status_dir():
    matches = sorted((ROOT / "logs").glob("readme_validation_full_*"))
    return matches[-1] if matches else None


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def number(value):
    return f"{float(value):.12e}" if np.isfinite(value) else ""


def build_specs():
    specs = []
    for problem in PROBLEMS:
        specs.append(
            {
                "family": "DEQGAN",
                "status_family": "deqgan",
                "problem": problem,
                "method": "deqgan",
                "pkey": problem,
                "trainer": "deqgan",
                "variant": "base",
            }
        )
    for problem in PROBLEMS:
        for method in METHODS:
            specs.append(
                {
                    "family": "NTK-GAN-RB",
                    "status_family": "ntk_rb",
                    "problem": problem,
                    "method": method,
                    "pkey": f"{problem}_ntk_{method}",
                    "trainer": "ntk",
                    "variant": "rb",
                }
            )
    for problem in PROBLEMS:
        for method in METHODS:
            specs.append(
                {
                    "family": "NTK-GAN-noRB",
                    "status_family": "ntk_norb",
                    "problem": problem,
                    "method": method,
                    "pkey": f"{problem}_ntk_{method}",
                    "trainer": "ntk",
                    "variant": "norb",
                }
            )
    specs.append(
        {
            "family": "Appendix H.3.2",
            "status_family": "lap1_ablation",
            "problem": "lap1",
            "method": "gan",
            "pkey": "lap1_ntk_gan_deqgan",
            "trainer": "ntk",
            "variant": "ablation",
        }
    )
    return specs


def read_runtime(status_dir):
    runtime = {}
    if status_dir is None:
        return runtime
    for path in sorted(status_dir.glob("status_cuda_*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                runtime[(row["family"], row["pkey"])] = row
    return runtime


def expected_flags(variant):
    if variant == "norb":
        return {
            "G_iters": 1,
            "D_iters": 1,
            "adaptive_G": False,
            "adaptive_D": False,
            "ntk": False,
            "energy": False,
        }
    if variant in ("rb", "ablation"):
        return {
            "G_iters": 20,
            "D_iters": 20,
            "adaptive_G": True,
            "adaptive_D": True,
            "ntk": True,
            "energy": True,
        }
    return {}


def checkpoint_is_finite(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("generator_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        return False
    return all(torch.isfinite(value).all().item() for value in state.values())


def load_first_metric(run_dir, filenames):
    """Load the preferred metric filename, with fallbacks for historical runs."""
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            array = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
            return filename, array
    return None, None


def collect(spec, runtime):
    source_config_path = CONFIG_DIR / f"{spec['pkey']}.yaml"
    with source_config_path.open() as handle:
        source_config = yaml.safe_load(handle)
    source_training = source_config["training"]

    if spec["variant"] == "norb":
        run_dir = RUNS_DIR / f"{RUN_STEMS[spec['problem']]}_NTK_none_1_1_{spec['method']}"
    else:
        run_dir = RUNS_DIR / source_training["dirname"]

    issues = []
    saved_config_path = run_dir / "config.yaml"
    if saved_config_path.exists():
        with saved_config_path.open() as handle:
            saved_config = yaml.safe_load(handle)
        training = saved_config["training"]
    else:
        training = source_training
        issues.append("missing config.yaml")

    expected = int(training["niters"])
    metric_candidates = (
        [
            ("val_mse.npy",),
            ("train_mse.npy",),
            ("G_loss.npy",),
            ("D_loss.npy",),
            ("rKr_gamma_loss.npy",),
            ("metrics_residual_energy.npy",),
        ]
        if spec["trainer"] == "deqgan"
        else [
            ("val_mse.npy", "mse_val.npy"),
            ("train_mse.npy", "mse_train.npy"),
            ("best_G_step.npy",),
            ("best_D_step.npy",),
            ("rKr_gamma_loss.npy", "loss_rKr_gamma.npy"),
            ("residual_energy_loss.npy", "loss_residual_energy.npy"),
        ]
    )
    arrays = {}
    for filenames in metric_candidates:
        preferred_name = filenames[0]
        loaded_name, array = load_first_metric(run_dir, filenames)
        if array is None:
            issues.append(f"missing {' or '.join(filenames)}")
            continue
        arrays[preferred_name] = array
        if len(array) != expected:
            issues.append(f"{loaded_name} length {len(array)} != {expected}")
        if not np.isfinite(array).all():
            issues.append(f"{loaded_name} has non-finite values")

    for name in ("plot_1.png", "plot3D.png"):
        if not (run_dir / name).exists():
            issues.append(f"missing {name}")

    if spec["trainer"] == "ntk":
        checkpoint = run_dir / "generator_final.pt"
        if not checkpoint.exists():
            issues.append("missing generator_final.pt")
        else:
            try:
                if not checkpoint_is_finite(checkpoint):
                    issues.append("invalid/non-finite generator checkpoint")
            except Exception as exc:
                issues.append(f"checkpoint load failed: {type(exc).__name__}")

    for key, wanted in expected_flags(spec["variant"]).items():
        if training.get(key) != wanted:
            issues.append(f"config {key}={training.get(key)!r}, expected {wanted!r}")

    status_row = runtime.get((spec["status_family"], spec["pkey"]), {})
    if status_row:
        if status_row.get("exit_code") != "0" or status_row.get("status") != "complete":
            issues.append(
                f"runner status={status_row.get('status')}, exit={status_row.get('exit_code')}"
            )
    else:
        issues.append("missing runner status row")

    val = arrays.get("val_mse.npy", np.asarray([]))
    train = arrays.get("train_mse.npy", np.asarray([]))
    return {
        "family": spec["family"],
        "problem": DISPLAY[spec["problem"]],
        "method": spec["method"].upper(),
        "pkey": spec["pkey"],
        "status": "complete" if not issues else "incomplete",
        "expected_epochs": expected,
        "recorded_epochs": len(val),
        "final_train_mse": number(train[-1]) if len(train) else "",
        "best_train_mse": number(train.min()) if len(train) else "",
        "final_val_mse": number(val[-1]) if len(val) else "",
        "best_val_mse": number(val.min()) if len(val) else "",
        "elapsed_seconds": status_row.get("elapsed_seconds", ""),
        "device": status_row.get("device", ""),
        "result_dir": rel(run_dir),
        "issues": "; ".join(issues),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, status_dir):
    complete = sum(row["status"] == "complete" for row in rows)
    families = []
    for family in ("DEQGAN", "NTK-GAN-RB", "NTK-GAN-noRB", "Appendix H.3.2"):
        selected = [row for row in rows if row["family"] == family]
        families.append((family, sum(r["status"] == "complete" for r in selected), len(selected)))

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    with path.open("w") as handle:
        handle.write("# README experiment validation\n\n")
        handle.write(f"- Overall: **{complete}/{len(rows)} complete**\n")
        handle.write(f"- Audited outer epochs: **{sum(int(r['recorded_epochs']) for r in rows):,}**\n")
        handle.write(f"- Python: `{platform.python_version()}`\n")
        handle.write(f"- PyTorch: `{torch.__version__}`\n")
        handle.write(f"- CUDA available during summary: `{torch.cuda.is_available()}`\n")
        handle.write(f"- GPUs: `{', '.join(gpu_names) if gpu_names else 'not detected'}`\n")
        handle.write(f"- Status source: `{rel(status_dir) if status_dir else 'not found'}`\n\n")
        handle.write("## Completion by family\n\n")
        handle.write("| Family | Complete | Total |\n|---|---:|---:|\n")
        for family, done, total in families:
            handle.write(f"| {family} | {done} | {total} |\n")
        handle.write("\n## Per-experiment results\n\n")
        handle.write(
            "| Family | PDE | Method | Status | Epochs | Final validation MSE | "
            "Best validation MSE | Runtime (s) |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['family']} | {row['problem']} | {row['method']} | "
                f"{row['status']} | {row['recorded_epochs']}/{row['expected_epochs']} | "
                f"{row['final_val_mse']} | {row['best_val_mse']} | "
                f"{row['elapsed_seconds']} |\n"
            )
        handle.write("\n## Audit criteria\n\n")
        handle.write(
            "A run is complete only when the runner exit status is zero; every required metric "
            "array has the configured length and finite values; both solution plots and the saved "
            "configuration exist; and, for NTK runs, the final checkpoint loads on CPU with a "
            "non-empty, finite generator state. Rollback switches are also checked against the "
            "README definition. The CSV contains result directories and any audit issues.\n"
        )


def main():
    args = parse_args()
    status_dir = args.status_dir
    if status_dir is not None and not status_dir.is_absolute():
        status_dir = ROOT / status_dir
    if status_dir is None:
        status_dir = latest_status_dir()
    runtime = read_runtime(status_dir)
    rows = [collect(spec, runtime) for spec in build_specs()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.name}.csv"
    md_path = RESULTS_DIR / f"{args.name}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, status_dir)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"Audited {complete}/{len(rows)} experiments as complete")
    print(f"Wrote {rel(csv_path)}")
    print(f"Wrote {rel(md_path)}")
    return 0 if complete == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
