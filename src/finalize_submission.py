"""Populate the real baseline runtime in docs and verify Unit 3 deliverables."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    data_path = ROOT / "results/performance/baseline_full_runtime.json"
    if not data_path.is_file():
        print("MISSING: baseline timing. Run python -m src.baseline_timing first.")
        return 1
    record = json.loads(data_path.read_text(encoding="utf-8"))
    seconds = float(record["baseline_full_runtime_seconds"])
    if record.get("total_replicates") != 80 or record.get("n") != 400 or record.get("reps_per_scenario") != 40:
        print("ERROR: baseline timing was not the full default 2 x 40, n=400 experiment.")
        return 1
    docpath = ROOT / "docs/BASELINE.md"
    doc = docpath.read_text(encoding="utf-8")
    replacement = ("**Local full-run timing:** " + f"{seconds:.3f} seconds " +
                   "for 2 scenarios x 40 replicates (n=400, 121 grid points, cap 350), " +
                   "measured on the local machine using `python -m src.baseline_timing`, " +
                   "including generation, EM fitting, posterior means, metrics and CSV writing " +
                   "(excluding analysis/figures). Saved in `results/performance/baseline_full_runtime.json`.")
    doc, count = re.subn(r"\*\*Local full-run timing:\*\*[^\n]*", replacement, doc)
    if count != 1:
        print("ERROR: runtime paragraph not found in docs/BASELINE.md")
        return 1
    docpath.write_text(doc,encoding="utf-8")
    required = ["README.md", "ADEMP.md", "ANALYSIS.md", "Makefile", "requirements.txt",
                "docs/BASELINE.md", "docs/OPTIMIZATION.md", "src/model.py",
                "src/model_baseline.py", "src/simulation.py", "src/performance.py",
                "tests/test_regression.py", "results/raw/simulation.csv", "results/raw/summary.csv",
                "results/performance/benchmark.csv", "results/performance/complexity.csv",
                "results/performance/complexity_slopes.csv", "results/performance/stability.csv",
                "results/performance/benchmark.png", "results/performance/complexity.png",
                "results/figures/mse_comparison.png", "results/performance/baseline_full_runtime.json"]
    missing = [p for p in required if not (ROOT / p).is_file()]
    print(f"Original Unit 2 full-run timing: {seconds:.3f} seconds")
    print("Documentation updated: docs/BASELINE.md")
    if missing:
        print("Missing files:", *missing, sep="\n - ")
        return 1
    print("Submission file check: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
