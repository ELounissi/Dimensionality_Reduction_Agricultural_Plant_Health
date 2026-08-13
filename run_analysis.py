#!/usr/bin/env python3
"""Reproduce the paper analysis or regenerate outputs from the shipped results."""
from __future__ import annotations

import argparse
import json

from analysis.workflow import (audit_protocol, generate_outputs, reproduce_all,
                               verify_outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/reproduced",
                        help="output directory for a full run")
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel worker processes; default is CPU count minus one")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of deterministic seeds; use 10 for the paper")
    parser.add_argument("--from-existing", action="store_true",
                        help="regenerate tables, tests, and figures from shipped results")
    args = parser.parse_args()

    if args.from_existing:
        target = "results"
        audit_protocol(target)
        generate_outputs(target)
        summary = verify_outputs(target, expected_seeds=10)
    else:
        summary = reproduce_all(args.results_dir, seeds=args.seeds, jobs=args.jobs)

    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit("RESULT_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
