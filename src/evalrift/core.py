from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Callable, Mapping


@dataclass(frozen=True)
class MCQ:
    question: str
    choices: tuple[str, ...]
    answer: int


Transform = Callable[[MCQ, int], MCQ]


def reorder(item: MCQ, seed: int) -> MCQ:
    idx = list(range(len(item.choices)))
    random.Random(seed).shuffle(idx)
    choices = tuple(item.choices[i] for i in idx)
    answer = idx.index(item.answer)
    return MCQ(item.question, choices, answer)


def whitespace(item: MCQ, seed: int) -> MCQ:
    pad = " " * (1 + seed % 3)
    return MCQ(pad + item.question + pad, item.choices, item.answer)


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


def accuracy(model: Callable[[MCQ], int], items: list[MCQ]) -> float:
    return sum(model(x) == x.answer for x in items) / max(1, len(items))


def _paired_drop_ci(
    base_correct: list[float],
    perturbed_correct: list[float],
    *,
    samples: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    if len(base_correct) != len(perturbed_correct):
        raise ValueError("paired arrays must have the same length")
    if not base_correct or samples <= 0:
        return (0.0, 0.0)

    differences = [a - b for a, b in zip(base_correct, perturbed_correct)]
    rng = random.Random(seed)
    n = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return (low, high)


def audit_suite(
    model: Callable[[MCQ], int],
    items: list[MCQ],
    *,
    rounds: int = 20,
    transforms: Mapping[str, Transform] | None = None,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    selected = dict(
        transforms
        or {
            "answer_order": reorder,
            "question_whitespace": whitespace,
        }
    )
    base_correct = [float(model(item) == item.answer) for item in items]
    base = sum(base_correct) / max(1, len(base_correct))

    report: dict[str, dict] = {}
    for transform_name, transform in selected.items():
        scores: list[float] = []
        per_item_hits = [0.0 for _ in items]

        for r in range(rounds):
            perturbed = [
                transform(item, stable_seed(item.question) + seed + r)
                for item in items
            ]
            round_correct = [float(model(item) == item.answer) for item in perturbed]
            scores.append(sum(round_correct) / max(1, len(round_correct)))
            for i, hit in enumerate(round_correct):
                per_item_hits[i] += hit

        perturbed_correct = [x / rounds for x in per_item_hits]
        mean = sum(scores) / len(scores)
        low, high = _paired_drop_ci(
            base_correct,
            perturbed_correct,
            samples=bootstrap_samples,
            seed=seed + stable_seed(transform_name),
        )
        worst = min(scores)

        report[transform_name] = {
            "perturbed_mean": mean,
            "worst_accuracy": worst,
            "max_drop": base - worst,
            "mean_drop": base - mean,
            "drop_ci_95": [low, high],
            "round_scores": scores,
        }

    return {"base_accuracy": base, "transforms": report}


def robustness(model: Callable[[MCQ], int], items: list[MCQ], rounds: int = 20) -> dict:
    result = audit_suite(
        model,
        items,
        rounds=rounds,
        transforms={"answer_order": reorder},
    )
    out = result["transforms"]["answer_order"]
    return {
        "base_accuracy": result["base_accuracy"],
        "perturbed_mean": out["perturbed_mean"],
        "worst_accuracy": out["worst_accuracy"],
        "max_drop": out["max_drop"],
    }


def relabel(item: MCQ, labels: tuple[str, ...] = ("A", "B", "C", "D")) -> dict:
    if len(labels) < len(item.choices):
        raise ValueError("not enough labels")
    return {
        "question": item.question,
        "choices": {labels[i]: c for i, c in enumerate(item.choices)},
        "answer": labels[item.answer],
    }


def ranking(models: dict[str, Callable[[MCQ], int]], items: list[MCQ]) -> list[str]:
    return sorted(models, key=lambda name: accuracy(models[name], items), reverse=True)


def ranking_instability(
    models: dict[str, Callable[[MCQ], int]],
    items: list[MCQ],
    rounds: int = 20,
) -> float:
    base = ranking(models, items)
    changes = 0
    for r in range(rounds):
        perturbed = [reorder(x, stable_seed(x.question) + r) for x in items]
        changes += ranking(models, perturbed) != base
    return changes / max(1, rounds)
