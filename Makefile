PYTHON = python
.PHONY: all simulate analyze figures test clean profile complexity benchmark parallel stability-check baseline-timing finalize
all: simulate analyze figures
simulate:
	$(PYTHON) -m src.simulation
analyze:
	$(PYTHON) -m src.analyze
figures:
	$(PYTHON) -m src.figures
test:
	$(PYTHON) -m pytest -q
profile:
	$(PYTHON) -m src.performance profile
complexity:
	$(PYTHON) -m src.performance complexity
benchmark:
	$(PYTHON) -m src.performance benchmark
parallel:
	$(PYTHON) -m src.simulation --workers 2
stability-check:
	$(PYTHON) -m src.performance stability-check
clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for sub in ('results/raw', 'results/figures', 'results/performance') for p in Path(sub).glob('*') if p.is_file()]"

baseline-timing:
	$(PYTHON) -m src.baseline_timing
finalize:
	$(PYTHON) -m src.finalize_submission
