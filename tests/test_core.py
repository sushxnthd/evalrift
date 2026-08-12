from evalrift.core import MCQ, audit_suite, reorder, robustness


def test_reorder_preserves_answer_text():
    x = MCQ("q", ("a", "b", "c"), 1)
    y = reorder(x, 4)
    assert y.choices[y.answer] == "b"


def test_sensitive_baseline_drops():
    items = [MCQ(str(i), ("ok", "x", "y", "z"), 0) for i in range(30)]
    assert robustness(lambda x: 0, items, 10)["max_drop"] > 0


def test_relabel_preserves_correct_text():
    from evalrift.core import relabel

    x = MCQ("q", ("x", "correct", "z"), 1)
    r = relabel(x)
    assert r["choices"][r["answer"]] == "correct"


def test_ranking_instability_bounded():
    from evalrift.core import ranking_instability

    items = [MCQ(str(i), ("ok", "x", "y", "z"), 0) for i in range(20)]
    v = ranking_instability({"first": lambda x: 0, "second": lambda x: 1}, items, 5)
    assert 0 <= v <= 1


def test_audit_suite_runs_multiple_semantic_transforms():
    items = [MCQ(str(i), ("ok", "x", "y", "z"), 0) for i in range(30)]
    report = audit_suite(lambda x: 0, items, rounds=8, bootstrap_samples=100, seed=7)
    assert set(report["transforms"]) == {"answer_order", "question_whitespace"}
    assert report["transforms"]["answer_order"]["mean_drop"] > 0
    assert report["transforms"]["question_whitespace"]["mean_drop"] == 0


def test_drop_interval_is_deterministic_and_bounded():
    items = [MCQ(str(i), ("ok", "x", "y", "z"), 0) for i in range(25)]
    a = audit_suite(lambda x: 0, items, rounds=6, bootstrap_samples=80, seed=3)
    b = audit_suite(lambda x: 0, items, rounds=6, bootstrap_samples=80, seed=3)
    assert a == b
    low, high = a["transforms"]["answer_order"]["drop_ci_95"]
    assert -1 <= low <= high <= 1
