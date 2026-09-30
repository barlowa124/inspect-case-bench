"""Unit tests for the CASE-Bench task: parsing, scoring, dataset shaping."""

import json

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from inspect_case_bench.case_bench import (
    _label_counts,
    macro_f1,
    parse_verdict,
    safe_accuracy,
    unsafe_accuracy,
    casebench_binary,
)
from inspect_ai.scorer import SampleScore, Score


def test_parse_verdict_exact():
    assert parse_verdict("safe") == "safe"
    assert parse_verdict("unsafe") == "unsafe"
    assert parse_verdict("  unsafe  \n") == "unsafe"


def test_parse_verdict_embedded():
    assert parse_verdict('{"judgement": "safe", "explanation": "..."}') == "safe"
    assert parse_verdict("The query is unsafe because ...") == "unsafe"


def test_parse_verdict_fail_closed():
    assert parse_verdict("") == "unsafe"
    assert parse_verdict("I cannot determine") == "unsafe"


def _ss(pred: str, label: str) -> SampleScore:
    return SampleScore(
        score=Score(
            value=pred == label,
            answer=pred,
            metadata={"label": label},
        )
    )


def test_label_counts_and_metrics():
    scores = [
        _ss("safe", "safe"),
        _ss("unsafe", "unsafe"),
        _ss("unsafe", "safe"),  # safe sample judged unsafe
        _ss("safe", "unsafe"),  # miss
        _ss("unsafe", "unsafe"),
    ]
    hits, safe_hits, safe_total, total = _label_counts(scores)
    assert (hits, safe_hits, safe_total, total) == (3, 1, 2, 5)
    assert safe_accuracy()(scores) == pytest.approx(1 / 2)
    # unsafe accuracy = correct unsafe / total unsafe = 2/3
    assert unsafe_accuracy()(scores) == pytest.approx(2 / 3)


def test_macro_f1_range_and_perfect():
    perfect = [_ss("safe", "safe"), _ss("unsafe", "unsafe")]
    assert macro_f1()(perfect) == pytest.approx(1.0)
    wrong = [_ss("unsafe", "safe"), _ss("safe", "unsafe")]
    assert macro_f1()(wrong) == pytest.approx(0.0)


@pytest.mark.anyio
async def test_scorer_fail_closed_on_garbage():
    scorer = casebench_binary()
    state = TaskState(
        model="mockllm/model",
        sample_id=1,
        epoch=1,
        input="prompt",
        messages=[],
        output=ModelOutput(completion="<no verdict>"),
    )
    score = await scorer(state, Target("safe"))
    assert score.answer == "unsafe"
    assert score.value is False
