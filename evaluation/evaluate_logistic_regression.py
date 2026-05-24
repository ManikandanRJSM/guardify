import pandas as pd
import joblib
from pathlib import Path
from training.helpers.GetEnv import GetEnv
from guardify.utils.ml_utils import MlUtils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def train(_env):
    df = pd.read_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")

    split = round(0.7 * len(df))
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    vectorizer = MlUtils.get_vectorizer_config()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, train_df["label"])

    acc = accuracy_score(test_df["label"], clf.predict(X_test)) * 100
    print(f"Accuracy: {acc:.2f}%")
    print("Confusion matrix:\n", confusion_matrix(test_df["label"], clf.predict(X_test)))


if __name__ == "__main__":
    _env = GetEnv.get_env_variables()
    train(_env)
