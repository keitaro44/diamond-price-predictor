# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict[str, float | str]:
    return {
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "SI2",
        "depth": 61.5,
        "table": 55.0,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    }


def test_make_inference(monkeypatch, create_data):

    def mock_get_predictions(_, data: pd.DataFrame) -> list[float]:
        for key, value in create_data.items():
            assert data[key].iloc[0] == value
        return [326.0]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"price": 326.0}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert filepath_and_data[1] == load_model(filepath_and_data[0])