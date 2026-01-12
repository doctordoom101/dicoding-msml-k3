import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
from contextlib import nullcontext  # <--- IMPORT BARU PENTING

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- KONFIGURASI ---
DAGSHUB_USER = "fandadefchristian"
DAGSHUB_REPO = "Fraud-Detection-Experiment"
EXPERIMENT_NAME = "Fraud_Detection_Experiment"

# Path Dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_preprocessing", "clean_card_transdata.csv")

# --- FUNGSI BANTUAN BARU ---
def get_mlflow_run():
    """
    Fungsi pintar untuk menentukan apakah harus membuat Run baru
    atau menggunakan Run yang sudah dibuat oleh GitHub Actions.
    """
    if mlflow.active_run():
        print(f"Active run detected (ID: {mlflow.active_run().info.run_id}). Menggunakan run yang ada.")
        # Jika sudah ada run aktif (dari CI pipeline), jangan start run baru.
        # Kembalikan 'nullcontext' yang tidak melakukan apa-apa.
        return nullcontext()
    else:
        print("Tidak ada active run. Memulai run baru secara manual.")
        # Jika dijalankan di laptop, set experiment dan start run baru.
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run(run_name="RandomForest_Manual_Log")

def train_model():
    print("=== Memulai Proses Training Model ===")
    
    # 1. Setup DagsHub (Agar Token Auth terbaca)
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    
    # (Hapus mlflow.set_experiment di sini, sudah dipindah ke fungsi bantuan)

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        return

    print(f"Loading data dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    TARGET_COLUMN = 'class' 
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Kolom target '{TARGET_COLUMN}' tidak ditemukan.")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai MLflow Run (MENGGUNAKAN FUNGSI BARU)
    # Gunakan 'get_mlflow_run()' sebagai pengganti 'mlflow.start_run()' langsung
    with get_mlflow_run():
        
        print("Training dimulai dalam konteks MLflow...")

        # --- A. Define Hyperparameters ---
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # --- B. Log Parameters ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "Random Forest Classifier")
        
        # --- C. Train Model ---
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- D. Metrics ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Hasil Training -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # --- Log Metrics ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- E. Artifacts ---
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plot_filename = "confusion_matrix.png"
        plt.savefig(plot_filename)
        plt.close()
        mlflow.log_artifact(plot_filename)

        # Report Text
        report = classification_report(y_test, y_pred)
        report_filename = "classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_filename)

        # Log Model
        print("Menyimpan model...")
        mlflow.sklearn.log_model(model, "model")

        # Cleanup
        if os.path.exists(plot_filename): os.remove(plot_filename)
        if os.path.exists(report_filename): os.remove(report_filename)

if __name__ == "__main__":
    train_model()