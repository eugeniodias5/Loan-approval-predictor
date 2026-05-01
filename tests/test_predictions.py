import pytest

from loan_approval_predictor.processing.data_management import load_data
from loan_approval_predictor.predict import predict


@pytest.fixture
def single_prediction():
    X, _ = load_data()

    # Slice X to get a single row
    single_X = X.iloc[0:1]

    return predict(single_X)


def test_single_prediction_not_none(single_prediction):
    assert single_prediction is not None, "The prediction should not be None."


def test_single_prediction_is_string(single_prediction):
    assert isinstance(single_prediction[0], str), "The prediction should be a string."
