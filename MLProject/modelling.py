import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- KONFIGURASI ---
# GANTI DENGAN USERNAME DAN NAMA REPO DAGSHUB ANDA
DAGSHUB_USER = "fandadefchristian"
DAGSHUB_REPO = "Fraud-Detection-Experiment"
EXPERIMENT_NAME = "Fraud_Detection_Experiment"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_preprocessing", "clean_card_transdata.csv")

def train_model():
    print("=== Memulai Proses Training Model ===")
    
    # 1. Setup DagsHub & MLflow Connection
    print(f"Menghubungkan ke DagsHub: {DAGSHUB_USER}/{DAGSHUB_REPO}...")
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File tidak ditemukan di {DATA_PATH}")
        return

    print(f"Loading data dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # --- PENTING: Sesuaikan nama kolom target di bawah ini ---
    # Jika di dataset Anda nama kolom targetnya 'is_fraud', ganti 'class' menjadi 'is_fraud'
    TARGET_COLUMN = 'fraud' 
    
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Kolom target '{TARGET_COLUMN}' tidak ditemukan di dataset.")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Mulai MLflow Run
    with mlflow.start_run(run_name="RandomForest_Manual_Log"):
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
        
        # Predict
        y_pred = model.predict(X_test)

        # --- D. Calculate Metrics ---
        # Gunakan average='weighted' karena data fraud biasanya imbalance
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Hasil Training -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # --- E. Log Metrics (Manual) ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- F. Log ARTIFACTS (Minimal 2 Tambahan) ---
        
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

        # --- G. Log Model ---
        print("Menyimpan model ke Remote Storage...")
        mlflow.sklearn.log_model(model, "model")
        
        # Bersihkan file temporary lokal
        if os.path.exists(plot_filename): os.remove(plot_filename)
        if os.path.exists(report_filename): os.remove(report_filename)

        print("=== Proses Selesai. Cek DagsHub untuk hasil eksperimen. ===")

if __name__ == "__main__":
    train_model()