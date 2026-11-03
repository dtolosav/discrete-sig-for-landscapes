.PHONY: test verify demo
test:
	python -m pytest -q tests
verify:
	python -m pip check
	python -c 'import fruits, pysiglib, numpy, scipy, sklearn'
demo:
	python scripts/run_revision.py --config configs/x1.json --scope pilot --output-dir results/demo
	python scripts/report_revision.py --run-dir results/demo
