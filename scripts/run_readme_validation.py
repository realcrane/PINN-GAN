#!/usr/bin/env python3
"""Run and audit the experiment matrix described by README.md."""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ("pos", "lap1", "lap2", "kle", "rea", "burv")
METHODS = ("gan", "lsgan", "wgan")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "full"), default="smoke"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("deqgan", "ntk_rb", "ntk_norb", "lap1_ablation"),
        default=("deqgan", "ntk_rb", "ntk_norb", "lap1_ablation"),
    )
    parser.add_argument(
        "--tag",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    return parser.parse_args()


def build_runs(args):
    runs = []
    if "deqgan" in args.families:
        runs.extend(("deqgan", problem, []) for problem in args.problems)
    if "ntk_rb" in args.families:
        runs.extend(
            ("ntk_rb", f"{problem}_ntk_{method}", [])
            for problem in args.problems
            for method in METHODS
        )
    if "ntk_norb" in args.families:
        runs.extend(
            ("ntk_norb", f"{problem}_ntk_{method}", ["--disable-rollback"])
            for problem in args.problems
            for method in METHODS
        )
    if "lap1_ablation" in args.families and "lap1" in args.problems:
        runs.append(("lap1_ablation", "lap1_ntk_gan_deqgan", []))
    return runs


def main():
    args = parse_args()
    log_dir = ROOT / "logs" / f"readme_validation_{args.mode}_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    device_label = args.device.replace(":", "_")
    status_path = log_dir / f"status_{device_label}.csv"
    fields = (
        "family",
        "pkey",
        "device",
        "mode",
        "start_utc",
        "elapsed_seconds",
        "exit_code",
        "status",
        "log_path",
    )

    rows = []
    for family, pkey, extra in build_runs(args):
        trainer = "deqgan" if family == "deqgan" else "ntk"
        command = [
            sys.executable,
            "-u",
            str(ROOT / "run_experiment.py"),
            "--trainer",
            trainer,
            "--pkey",
            pkey,
            "--device",
            args.device,
            *extra,
        ]
        if args.mode == "smoke":
            smoke_dir = Path("/tmp") / f"pinn_gan_smoke_{args.tag}_{family}_{pkey}"
            command.extend(("--niters", "1", "--dirname", str(smoke_dir)))

        log_path = log_dir / f"{family}__{pkey}.log"
        started = datetime.now(timezone.utc)
        print(f"START {family} {pkey} on {args.device}", flush=True)
        before = time.monotonic()
        with log_path.open("w") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.monotonic() - before
        status = "complete" if completed.returncode == 0 else "failed"
        print(
            f"{status.upper()} {family} {pkey} "
            f"exit={completed.returncode} elapsed={elapsed:.1f}s",
            flush=True,
        )
        rows.append(
            {
                "family": family,
                "pkey": pkey,
                "device": args.device,
                "mode": args.mode,
                "start_utc": started.isoformat(),
                "elapsed_seconds": f"{elapsed:.3f}",
                "exit_code": completed.returncode,
                "status": status,
                "log_path": str(log_path.relative_to(ROOT)),
            }
        )
        with status_path.open("w", newline="") as status_handle:
            writer = csv.DictWriter(status_handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    failures = sum(row["status"] != "complete" for row in rows)
    print(
        f"SUMMARY device={args.device} complete={len(rows) - failures}/"
        f"{len(rows)} status_csv={status_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
