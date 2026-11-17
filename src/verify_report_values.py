"""Check that narrative benchmark and complexity figures match saved CSV evidence."""
import csv
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    report = (root / "docs/OPTIMIZATION.md").read_text(encoding="utf-8")
    baseline = (root / "docs/BASELINE.md").read_text(encoding="utf-8")
    with (root / "results/performance/benchmark.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (root / "results/performance/complexity_slopes.csv").open(newline="") as f:
        slopes = list(csv.DictReader(f))
    for x in rows:
        line = f"| {x['scenario']} | {x['n']} | {float(x['baseline_s']):.6f} | {float(x['optimized_s']):.6f} | {float(x['speedup']):.2f}× |"
        assert line in report, f"Benchmark row mismatch: {x}"
    for x in slopes:
        value = f"{float(x['loglog_slope']):.4f}"
        assert value in report, f"Optimization slope mismatch: {x}"
        if x["version"] == "baseline":
            assert value in baseline, f"Baseline slope mismatch: {x}"
    print("Report values match current saved benchmark and complexity results.")

if __name__ == "__main__":
    main()
