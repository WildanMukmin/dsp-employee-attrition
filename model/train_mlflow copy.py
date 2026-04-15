import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

load_dotenv()

def run_rf_model_mlflow(df):
    # =========================
    # SET MLFLOW DAGSHUB
    # =========================
    uri_dagshub = "https://dagshub.com/WildanMukmin/dsp-employee-attrition.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not username or not password:
        raise ValueError("Username/Password MLflow tidak ditemukan di .env")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    # =========================
    # SET EXPERIMENT
    # =========================
    experiment_name = "attrition_prediction"
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    print(f"Menggunakan experiment: {experiment_name} (ID: {experiment_id})")

    # =========================
    # PREPROCESSING
    # =========================
    print("Membersihkan data...")
    df_clean = df.dropna(subset=["FinalAttrition"])

    y = df_clean["FinalAttrition"]
    X = df_clean[["Age", "MonthlyIncome", "JobLevel", "YearsAtCompany"]]

    # Jika ada kategorikal, amanin
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # =========================
    # TRAINING + LOGGING
    # =========================
    mlflow.sklearn.autolog(log_models=False)  
    # ❗ penting: biar ga double log model

    with mlflow.start_run(run_name="rf-default-model") as run:

        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)

        # Metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)

        # =========================
        # LOG MODEL
        # =========================
        signature = infer_signature(X_train, model_rf.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model_rf,
            artifact_path="model",  # ❗ bukan "name"
            registered_model_name="rf_model_attrition",
            signature=signature,
            input_example=X_train.head(1)
        )

        print("\n--- Proses Selesai ---")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print("Cek di DagsHub MLflow UI 🚀")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    dataset_path = "data/employe_data.csv"

    if os.path.exists(dataset_path):
        print("Memuat dataset...")
        df = pd.read_csv(dataset_path)
        run_rf_model_mlflow(df)
    else:
        print(f"File tidak ditemukan: {dataset_path}")