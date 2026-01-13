import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from contextlib import nullcontext 

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

def get_mlflow_run():
    """
    Logika pintar untuk mendeteksi environment.
    """
    if mlflow.active_run():
        print(f"Active run detected (ID: {mlflow.active_run().info.run_id}).")
        # Kita set tag di sini untuk memastikan run di CI punya nama yang jelas
        mlflow.set_tag("mlflow.runName", "RandomForest_CI_Pipeline")
        return nullcontext()
    else:
        print("Tidak ada active run. Memulai run baru secara manual.")
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run(run_name="RandomForest_Manual_Run")

def run_tuning():
    print("=== Memulai Training ===")

    # 1. Setup DagsHub Auth Saja
    # PENTING: mlflow=False agar tidak merusak konfigurasi tracking dari GitHub Actions
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=False)
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)
    TARGET_COLUMN = 'fraud' 
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target '{TARGET_COLUMN}' tidak ada.")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai Run
    with get_mlflow_run():
        print("Training dimulai...")
        
        # Params
        n_estimators = 50
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("tuning_method", "Manual_Fast")

        # Train
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Eval
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Hasil Training -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # --- Log Metrics (Manual) ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Artifacts (Gambar & Txt)
        print("Membuat Artifacts Visual...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title(f'Confusion Matrix (F1: {f1:.2f})')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # --- BAGIAN KRUSIAL (PENYIMPANAN MODEL) ---
        print("Menyimpan model...")
        
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_test, y_pred)

        # Langkah 1: Hapus folder lokal lama jika ada (biar bersih)
        local_model_path = "model_output"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)

        # Langkah 2: Simpan model ke folder LOKAL dulu (bukan langsung upload)
        # Ini memastikan struktur model valid sebelum dikirim
        mlflow.sklearn.save_model(
            sk_model=model,
            path=local_model_path,
            signature=signature,
            pip_requirements=["scikit-learn", "pandas", "numpy", "mlflow", "dagshub"]
        )
        print(f"Model berhasil disimpan secara lokal di: {local_model_path}")

        # Langkah 3: Upload folder lokal tersebut sebagai Artifact
        mlflow.log_artifacts(local_dir=local_model_path, artifact_path="model")
        print("Folder model berhasil di-upload ke DagsHub.")

        # Cleanup Lokal
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
        if os.path.exists("classification_report.txt"): os.remove("classification_report.txt")
        # Folder local_model_path biarkan saja, akan hilang saat container mati

    print("=== Selesai ===")

if __name__ == "__main__":
    run_tuning()