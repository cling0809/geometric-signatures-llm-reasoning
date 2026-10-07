import math

import pytest

from geoprobe.datasets.gsm8k import is_correct, parse_gold, parse_predicted


def test_parse_gold_simple():
    txt = "Janet had 3 apples. ... #### 7"
    assert parse_gold(txt) == 7.0


def test_parse_gold_comma_and_decimal():
    assert parse_gold("... #### 1,234.5") == 1234.5
    assert parse_gold("... #### -42") == -42.0


def test_parse_gold_missing():
    with pytest.raises(ValueError):
        parse_gold("no answer marker here")


def test_parse_predicted_boxed():
    assert parse_predicted("the answer is \\boxed{42}") == 42.0


def test_parse_predicted_boxed_with_commas():
    assert parse_predicted("\\boxed{1,000}") == 1000.0


def test_parse_predicted_falls_back_to_last_number():
    assert parse_predicted("Step 1: 10. Step 2: result is 25") == 25.0


def test_parse_predicted_ignores_synthetic_next_problem():
    txt = (
        "The profit is $70,000. Therefore, Josh made a profit of $70,000.\n\n"
        "Problem: A company has revenue $100,000 and cost $20,000."
    )
    assert parse_predicted(txt) == 70000.0


def test_parse_predicted_none_when_no_number():
    assert parse_predicted("I don't know") is None


def test_parse_predicted_rejects_non_finite():
    # A literal huge number (1 followed by 400 zeros) overflows float() -> inf.
    # We should treat that as no valid prediction (this is the real Sample-6 bug).
    huge = "1" + "0" * 400
    assert parse_predicted(f"the answer is {huge}") is None
    assert parse_predicted(f"\\boxed{{{huge}}}") is None


def test_is_correct_exact_and_tolerance():
    assert is_correct(7.0, 7.0)
    assert is_correct(7.00001, 7.0)
    assert not is_correct(7.1, 7.0)
    assert not is_correct(None, 7.0)
