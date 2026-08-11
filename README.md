# evalrift

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF)](.github/workflows/ci.yml) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


Benchmark fragility testing for multiple-choice evaluations. `evalrift` asks a simple question: does a reported score survive harmless changes to presentation?

```bash
pip install -e .
evalrift examples/sample.json --rounds 30
```

## Current perturbation

v0.1 focuses on answer-order randomization because it has a clean invariance: the correct answer text is unchanged while its position moves. The library preserves labels automatically and reports base accuracy, perturbed mean, worst-case accuracy and maximum drop.

## Benchmark

The included benchmark intentionally evaluates an answer-position-biased baseline. `python benchmarks/run.py` demonstrates the expected collapse under reordering and writes the measured result to `benchmarks/results.json`.

## Why narrow first

A perturbation suite is only useful when each transformation has an explicit semantic contract. Future transforms will be added with preservation tests rather than as arbitrary prompt noise.

## Roadmap

- label renaming and option-format transforms
- paired bootstrap confidence intervals
- model ranking instability across multiple systems
- adapters for common benchmark JSONL formats

MIT licensed.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the data flow and design boundaries.
