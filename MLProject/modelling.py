import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- KONFIGURASI ---
# Sesuaikan dengan config Anda
DAGSHUB_USER = "fandadefchristian"
DAGSHUB_REPO = "Fraud-Detection-Experiment"
EXPERIMENT_NAME = "Fraud_Detection_Experiment"

# Path Dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_preprocessing", "clean_card_transdata.csv")

def get_mlflow_run():
    """
    Mendeteksi apakah script dijalankan oleh 'mlflow run' (GitHub Actions)
    atau dijalankan manual (Laptop).
    """
    if mlflow.active_run():
        print(f"Active run detected (ID: {mlflow.active_run().info.run_id}). Menggunakan run dari CI.")
        
        # Opsional: Ubah nama run di CI agar terlihat rapi di DagsHub
        mlflow.set_tag("mlflow.runName", "RandomForest_CI_Pipeline")
        
        return nullcontext()
    else:
        print("Tidak ada active run. Memulai run baru secara manual.")
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run(run_name="RandomForest_Manual_Run")

def run_tuning():
    # 1. Setup DagsHub
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    
    # (HAPUS mlflow.set_experiment DISINI, SUDAH DITANGANI get_mlflow_run)

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

    # 3. Param Grid
    param_grid = {
        'n_estimators': [50, 100],      
        'max_depth': [10, 20],
        'class_weight': ['balanced']
    }

    # 4. Mulai MLflow Run 
    with get_mlflow_run():
        print("MLflow Run dimulai...")

        # --- A. Define Hyperparameters ---
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # --- B. Log Parameters (Manual) ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "Random Forest Classifier")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # --- C. Train Model ---
        print("Sedang melatih model...")
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth, 
                                       random_state=random_state)
        model.fit(X_train, y_train)
   
        # B. Evaluasi
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # C. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # D. Log Artifacts
        # Artifact 1: Confusion Matrix Plot (Gambar)
        print("Membuat Artifact 1: Confusion Matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan ke file lokal sementara
        plot_filename = "confusion_matrix.png"
        plt.savefig(plot_filename)
        plt.close() # Tutup plot agar tidak menumpuk di memori
        
        # Upload ke MLflow
        mlflow.log_artifact(plot_filename)

        # Artifact 2: Classification Report (Text File)
        print("Membuat Artifact 2: Classification Report...")
        report = classification_report(y_test, y_pred)
        report_filename = "classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        
        # Upload ke MLflow
        mlflow.log_artifact(report_filename)

        # E. Log Model Terbaik
        print("Menyimpan model terbaik ke DagsHub...")
        
        # DEFINISIKAN SIGNATURE (Opsional tapi disarankan agar MLflow tidak bingung)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_test, y_pred)

        # LOG MODEL DENGAN KONFIGURASI MANUAL
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            pip_requirements=["scikit-learn", "pandas", "numpy", "mlflow", "dagshub"]
        )

        # Cleanup
        if os.path.exists(plot_filename): os.remove(plot_filename)
        if os.path.exists(report_filename): os.remove(report_filename)

    print("=== Tuning Selesai. ===")

if __name__ == "__main__":
    run_tuning()