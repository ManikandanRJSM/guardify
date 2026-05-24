import joblib
from pathlib import Path

from guardify.guardrails.base import GuardrailBase

_MODEL_DIR = Path(__file__).parent.parent / "models"


class LogisticRegressionGuardrail(GuardrailBase):

    def __init__(self):
        self._clf = joblib.load(_MODEL_DIR / "lg_ip_guardrails.pkl")
        self._vectorizer = joblib.load(_MODEL_DIR / "ig_ip_vectorizer.pkl")

    def predict(self, text: str) -> dict:
        X = self._vectorizer.transform([text])
        label = int(self._clf.predict(X)[0])
        score = float(self._clf.predict_proba(X).max())
        return {
            "label": "injection" if label == 1 else "safe",
            "score": score,
        }
