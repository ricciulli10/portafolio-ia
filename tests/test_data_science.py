import pytest

from credit_risk.experiment import rejection_rate, sample_size, two_proportion_test
from credit_risk.sql_analysis import run


def test_sql_queries_cover_the_portfolio():
    results = run()
    assert results["default_rate_by_checking"]["loans"].sum() == 1000
    assert results["default_rate_by_duration_quartile"]["loans"].sum() == 1000
    share = results["riskiest_purposes_with_volume"]["exposure_share"].sum()
    assert share == pytest.approx(100, abs=0.5)


def test_two_proportion_test_matches_hand_calculation():
    result = two_proportion_test(100, 1000, 130, 1000)
    assert result["lift"] == 0.03
    assert result["p_value"] == pytest.approx(0.0355, abs=0.001)


def test_no_difference_gives_p_value_of_one():
    assert two_proportion_test(50, 100, 50, 100)["p_value"] == 1.0


def test_sample_size_grows_for_smaller_effects():
    assert sample_size(0.5, 0.01) > sample_size(0.5, 0.05)


def test_false_positive_rate_is_near_alpha():
    assert rejection_rate(0.3, 0.3, 500, 2000, 0) == pytest.approx(0.05, abs=0.015)
