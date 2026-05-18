from helpers.GetEnv import GetEnv
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd



def get_model_evaluation_metrics():
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Unsafe'], yticklabels=['Safe', 'Unsafe'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":

    _env = GetEnv.get_env_variables()

    df = pd.read_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")
    model_path = f"{_env['DATA_LAKE_PATH']}/model/lg_ip_guardrails.pkl"
    vectorizer_path = f"{_env['DATA_LAKE_PATH']}/model/vectorizer.pkl"

    mode = 'train_all' if _env['APP_ENV'] is not None and _env['APP_ENV'] == 'production' else 'dev'

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    if mode == 'dev':
        train_count = round((70/100)*len(df))
        test_count = len(df)-train_count

        test_dataset = df.iloc[train_count:(train_count+test_count)]
        X_test = vectorizer.transform(test_dataset['text'])
        y_test = test_dataset['label']

        # Prediction on dataset
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"Logistic Regression model accuracy: {acc:.2f}%")

        get_model_evaluation_metrics(y_test = y_test, y_pred = y_pred)

    else:
        pass
