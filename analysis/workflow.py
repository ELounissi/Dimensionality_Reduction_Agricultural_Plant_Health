"""Shared reproducibility workflow for the command-line script and notebook."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "analysis" / script), *map(str, arguments)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def audit_protocol(results_dir: str | Path = "results") -> Path:
    """Verify deterministic disjoint splits and training-only preprocessing."""
    root = PROJECT_ROOT / results_dir
    output = root / "manifests" / "protocol_audit.json"
    _run("protocol_audit.py", "--out", str(output))
    return output


def run_models(results_dir: str | Path = "results/reproduced", *, seeds: int = 10,
               jobs: int | None = None) -> Path:
    """Run the full reported model grid and save every run-level result."""
    root = PROJECT_ROOT / results_dir
    workers = jobs if jobs is not None else min(8, max(1, (os.cpu_count() or 2) - 1))
    _run("parallel_runner.py", "--seeds", str(seeds), "--seq-lens", "1", "7",
         "--n-jobs", str(workers), "--out", str(root))
    return root


def generate_outputs(results_dir: str | Path = "results") -> Path:
    """Generate statistical tests, tables, figures, and timing summaries."""
    root = PROJECT_ROOT / results_dir
    _run("statistics.py", "--results", str(root), "--split", "random")
    _run("reporting.py", "--results", str(root), "--split", "random")
    return root


def verify_outputs(results_dir: str | Path = "results", *, expected_seeds: int = 10) -> dict:
    """Check completeness and uniqueness of a generated result directory."""
    root = PROJECT_ROOT / results_dir
    runs = pd.read_csv(root / "runs.csv")
    aggregate = pd.read_csv(root / "aggregated.csv")
    keys = ["split", "task", "crop", "dr", "model", "seed", "seq_len"]
    figures = sorted((root / "figures").glob("*.png"))
    checks = {
        "run_rows": len(runs),
        "aggregate_rows": len(aggregate),
        "seeds": sorted(int(seed) for seed in runs.seed.unique()),
        "duplicate_result_keys": int(runs.duplicated(keys).sum()),
        "missing_scores": int(runs.score.isna().sum()),
        "figure_files": [path.name for path in figures],
        "statistical_tests_present": (root / "statistical_tests.csv").is_file(),
        "pairing_table_present": (root / "pairing_win_loss.csv").is_file(),
    }
    expected_seed_values = list(range(expected_seeds))
    checks["status"] = "PASS" if (
        checks["seeds"] == expected_seed_values
        and checks["duplicate_result_keys"] == 0
        and checks["missing_scores"] == 0
        and len(figures) == 6
        and checks["statistical_tests_present"]
        and checks["pairing_table_present"]
    ) else "FAIL"
    return checks


def reproduce_all(results_dir: str | Path = "results/reproduced", *, seeds: int = 10,
                  jobs: int | None = None) -> dict:
    """Run audit, model fitting, statistics, figures, and final verification."""
    audit_protocol(results_dir)
    run_models(results_dir, seeds=seeds, jobs=jobs)
    generate_outputs(results_dir)
    return verify_outputs(results_dir, expected_seeds=seeds)
