import os
import subprocess
import sys
import pickle


def test_train_script_runs_successfully():
    """Verify train.py executes successfully."""

    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Accuracy:" in result.stdout
    assert "AUC-ROC:" in result.stdout
    assert os.path.exists("models/churn_model.pkl")


def test_model_can_be_loaded():
    """Verify saved model can be loaded."""

    with open("models/churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    assert model is not None
    assert hasattr(model, "predict")