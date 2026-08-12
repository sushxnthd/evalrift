# evalrift

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF)](.github/workflows/ci.yml) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live demo](https://sushxnthd.github.io/evalrift/) | [Architecture](docs/architecture.md) | [Benchmarks](benchmarks/results.json)

Benchmark fragility testing for multiple-choice evaluations. `evalrift` checks whether a reported score survives presentation changes that should not alter the task.

```bash
pip install -e .
evalrift examples/sample.json --rounds 30
```

## Audit suites

Version 0.2 adds a transform suite instead of treating answer-order randomization as the only diagnostic. The default suite currently checks answer order and harmless question whitespace.

```python
from evalrift.core import audit_suite

report = audit_suite(model, items, rounds=30, seed=7)
```

Each transform reports:

- perturbed mean accuracy
- worst observed accuracy
- mean and maximum score drop
- per-round scores
- a paired bootstrap 95 percent interval for the mean drop

Custom transforms can be passed as a mapping of names to `MCQ, seed -> MCQ` callables. That keeps the semantic contract explicit and testable.

## Compatibility

The original `robustness()` function remains available and still evaluates answer-order sensitivity, so existing callers do not need to change immediately.

## Benchmark

The included benchmark intentionally evaluates an answer-position-biased baseline. `python benchmarks/run.py` demonstrates the expected collapse under reordering and writes the measured result to `benchmarks/results.json`.

## Scope

A perturbation audit can show that an evaluation is fragile. It cannot by itself explain why a model learned that fragility or prove that every transform preserves meaning for every dataset.

## Roadmap

- label renaming as a first-class transform
- common benchmark JSONL adapters
- multi-model rank concordance summaries
- HTML audit reports

MIT licensed.
