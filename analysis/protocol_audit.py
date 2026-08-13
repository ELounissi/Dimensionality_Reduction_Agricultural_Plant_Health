#!/usr/bin/env python3
"""Fail-fast checks for the plant-level evaluation boundary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dr_agri.data import CROPS, build_sequences, load_crop
from dr_agri.evaluate import _fit_view, _outer_split, prepare_neural_view


def _assert_disjoint(*parts: np.ndarray) -> None:
    sets = [set(np.asarray(p).tolist()) for p in parts]
    for i, left in enumerate(sets):
        for right in sets[i + 1:]:
            if left.intersection(right):
                raise AssertionError("row overlap across partitions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/manifests/protocol_audit.json")
    args = parser.parse_args()
    checks = []
    for task in ("classification", "regression"):
        for crop, cfg in CROPS.items():
            data = load_crop(crop, task)
            forbidden = {cfg["target"], *cfg["other_targets"]}
            leaked_columns = sorted(forbidden.intersection(data.feature_names))
            if leaked_columns:
                raise AssertionError(f"target predictors in {crop}: {leaked_columns}")
            sequences = build_sequences(data, 1)
            for seed in range(10):
                tr_a, te_a = _outer_split(data.y, task, seed)
                tr_b, te_b = _outer_split(data.y, task, seed)
                if not np.array_equal(tr_a, tr_b) or not np.array_equal(te_a, te_b):
                    raise AssertionError(f"non-deterministic outer split in {crop}/{task}/{seed}")
                _assert_disjoint(tr_a, te_a)
                if len(tr_a) + len(te_a) != len(data.y):
                    raise AssertionError("outer split does not cover every observation once")
                if task == "classification":
                    if np.unique(data.y[tr_a]).size != 2 or np.unique(data.y[te_a]).size != 2:
                        raise AssertionError(f"one-class outer split in {crop}/{seed}")

                view, _, sizes = prepare_neural_view(data, sequences, "No Reduction", seed)
                _assert_disjoint(view["indices"]["train"], view["indices"]["val"],
                                 view["indices"]["test"])
                if sum(len(view["indices"][p]) for p in ("train", "val", "test")) != len(data.y):
                    raise AssertionError("neural partitions do not cover every observation once")
                expected = (len(data.y) - len(te_a), len(te_a))
                if sizes[0] + sizes[1] != expected[0] or sizes[2] != expected[1]:
                    raise AssertionError(f"invalid 70/10/20 sizes in {crop}/{task}/{seed}: {sizes}")
                fractions = np.asarray(sizes, dtype=float) / len(data.y)
                if np.max(np.abs(fractions - np.array([0.70, 0.10, 0.20]))) > 0.01:
                    raise AssertionError(f"invalid 70/10/20 fractions in {crop}/{task}/{seed}: {fractions}")
                if task == "classification":
                    for partition in ("train", "val", "test"):
                        if np.unique(view[partition][1]).size != 2:
                            raise AssertionError(f"one-class neural {partition} in {crop}/{seed}")

                # A held-out block may be transformed but cannot alter the fitted
                # training representation. Changing it must leave Z_train fixed.
                X_fit = data.X[tr_a]
                X_test = data.X[te_a]
                z1, _ = _fit_view(X_fit, (X_test,), "PCA", data, seed)
                changed_test = np.nan_to_num(X_test, nan=0.0) + 1000.0
                z2, _ = _fit_view(X_fit, (changed_test,), "PCA", data, seed)
                if not np.allclose(z1, z2):
                    raise AssertionError("test data changed fitted preprocessing or PCA")

                checks.append(dict(crop=crop, task=task, seed=seed,
                                   classical_train=len(tr_a), classical_test=len(te_a),
                                   neural_train=sizes[0], neural_validation=sizes[1],
                                   neural_test=sizes[2], row_overlap=0,
                                   leaked_columns=[], context_columns=[c for c in ("FarmID", "Plant_ID", "Date")
                                                                        if c in data.feature_names],
                                   deterministic=True,
                                   training_only_preprocessing=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"state": "PASS", "checks": checks}, indent=2))
    print(f"PROTOCOL_AUDIT_PASS {len(checks)} split checks -> {out}")


if __name__ == "__main__":
    main()
