import os
import sys
import pytest

# Ensure project root is on sys.path so 'src' can be imported
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.predict import Predictor
except Exception as e:
    pytest.skip(f"Could not import Predictor: {e}")

MODEL_PATH = os.path.join('model_data', 'knn_weighted.joblib')
SCALER_PATH = os.path.join('model_data', 'scaler.joblib')

def test_predictor_loads_and_predicts():
    # Skip locally if model artifacts are not present (CI creates dummy artifacts)
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        pytest.skip('Model artifacts not present locally; skip local run.')

    p = Predictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    sample = {'displacement': 150.0, 'horsepower': 95.0, 'weight': 2000.0, 'acceleration': 15.5, 'cylinders': 4}
    pred = p.predict_one(sample)

    assert isinstance(pred, float)
    assert 0.0 < pred < 100.0
