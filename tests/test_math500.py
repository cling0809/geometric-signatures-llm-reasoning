from geoprobe.datasets.math500 import (
    is_correct_math500,
    parse_predicted_math500,
)


def test_parse_predicted_picks_last_boxed():
    txt = "First try: \\boxed{1} but actually \\boxed{42}"
    assert parse_predicted_math500(txt) == "42"


def test_parse_predicted_handles_fraction():
    txt = "The answer is \\boxed{\\frac{1}{2}}"
    # We allow the inner expression as a string
    assert parse_predicted_math500(txt) == "\\frac{1}{2}"


def test_parse_predicted_falls_back_to_last_number():
    txt = "Step 1: 3, step 2: 7. So the answer is 42."
    assert parse_predicted_math500(txt) == "42"


def test_parse_predicted_none_when_no_match():
    assert parse_predicted_math500("just words here") is None


def test_is_correct_float_match():
    assert is_correct_math500("42", "42")
    assert is_correct_math500("42.00001", "42")
    assert not is_correct_math500("43", "42")


def test_is_correct_string_match():
    # Symbolic expressions: only string equality (post-normalization) applies
    assert is_correct_math500("\\frac{1}{2}", "\\frac{1}{2}")
    assert is_correct_math500("x+1", "x + 1")  # whitespace normalized
    assert not is_correct_math500("x+1", "x+2")


def test_is_correct_handles_outer_boxed():
    assert is_correct_math500("\\boxed{42}", "42")
    assert is_correct_math500("42", "\\boxed{42}")


def test_is_correct_none_pred():
    assert not is_correct_math500(None, "42")


def test_symbolic_math500_grader_handles_equivalent_latex_and_decimal():
    from geoprobe.datasets.math500 import is_correct_math500_symbolic

    assert is_correct_math500_symbolic("Therefore the result is \\boxed{0.5}.", r"\frac{1}{2}")
    assert is_correct_math500_symbolic("Final: \\boxed{x^2 + 2x + 1}", r"(x+1)^2")
    assert not is_correct_math500_symbolic("Final: \\boxed{0.6}", r"\frac{1}{2}")


def test_symbolic_math500_grader_uses_complete_completion_not_legacy_last_number():
    from geoprobe.datasets.math500 import is_correct_math500_symbolic

    completion = "First I considered 7, but the final answer is \\boxed{42}."
    assert is_correct_math500_symbolic(completion, "42")


def test_symbolic_math500_grader_accepts_already_boxed_gold():
    from geoprobe.datasets.math500 import is_correct_math500_symbolic

    assert is_correct_math500_symbolic("Final: \\boxed{42}", r"\boxed{42}")
