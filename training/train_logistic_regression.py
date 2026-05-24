import pandas as pd
import joblib
from pathlib import Path
from .helpers.GetEnv import GetEnv
from guardify.utils.ml_utils import MlUtils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def train(_env):
    train_df = pd.read_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")

    vectorizer = MlUtils.get_vectorizer_config()
    X_train = vectorizer.fit_transform(train_df["text"])
    Y_train = train_df["label"] # Lables

    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, Y_train)

    model_dir = Path(_env["MODEL_PATH"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "lg_ip_guardrails.pkl")
    joblib.dump(vectorizer, model_dir / "ig_ip_vectorizer.pkl")
    print("Models saved.")


if __name__ == "__main__":
    _env = GetEnv.get_env_variables()
    train(_env)
