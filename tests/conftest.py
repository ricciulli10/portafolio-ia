import pytest

from credit_risk.data import load_credit
from credit_risk.model import MODEL_PATH, CreditModel, candidates, reference_row


@pytest.fixture(scope="session")
def credit_model() -> CreditModel:
    """Use the trained model if present, otherwise fit a quick one so CI stays self-contained."""
    if MODEL_PATH.exists():
        return CreditModel.load()
    X, y = load_credit()
    pipe = candidates(X)["logistic_regression"].fit(X, y)
    model = CreditModel(pipe, 0.18, reference_row(X), "test")
    model.save()
    return model
