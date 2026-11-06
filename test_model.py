
import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("model.pkl")
    sample = joblib.load("sample_input.pkl")

    predictions = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert abs(probabilities.sum() - 1.0) < 1e-6
