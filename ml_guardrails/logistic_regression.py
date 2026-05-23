import pandas as pd
import time
from helpers.GetEnv import GetEnv
import joblib
from ml_guardrails.Utils.MlUtils import MlUtils
from pathlib import Path

def train_with_logistic_regression(**kwargs):

    from sklearn.linear_model import LogisticRegression

    df = kwargs['df']
    _env = GetEnv.get_env_variables()

    path = Path(f"{_env['MODEL_PATH']}/logistic_regression")
    path.mkdir(parents=True, exist_ok=True)

    model_path = f"{_env['MODEL_PATH']}/lg_ip_guardrails.pkl"
    vectorizer_path = f"{_env['MODEL_PATH']}/ig_ip_vectorizer.pkl"

    vectorizer = MlUtils.get_vectorizer_config()

    X_train = vectorizer.fit_transform(df['text'])
    y_train = df['label']


    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Training completed.")

if __name__ == "__main__":

    _env = GetEnv.get_env_variables()

    df = pd.read_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")

    train_with_logistic_regression(df = df)
