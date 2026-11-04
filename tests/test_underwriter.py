import pytest

from underwriter.domain import BANNED_REASONING
from underwriter.evals import SCENARIOS, make_world


@pytest.fixture(scope="module")
def world(credit_model):
    return make_world()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario(world, scenario):
    passed, detail = scenario.run(world)
    assert passed, detail


@pytest.mark.parametrize("word", ["young", "Married", "foreign worker", "widowed"])
def test_protected_terms_detected(word):
    assert BANNED_REASONING.search(f"Declined because {word}.")


@pytest.mark.parametrize("sentence", ["The manager called.", "Mortgage is small.", "Stage two."])
def test_common_words_are_not_flagged(sentence):
    assert not BANNED_REASONING.search(sentence)
