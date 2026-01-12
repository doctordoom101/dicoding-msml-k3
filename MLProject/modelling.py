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

def run_tuning():
    print("=== Memulai Hyperparameter Tuning ===")

    # 1. Setup DagsHub
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File {DATA_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # --- SESUAIKAN KOLOM TARGET ---
    TARGET_COLUMN = 'fraud' 
    
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target '{TARGET_COLUMN}' tidak ada.")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Definisi Grid Parameter (Skenario Eksperimen)
    param_grid = {
        'n_estimators': [50, 100, 200],      # Jumlah pohon
        'max_depth': [10, 20, None],         # Kedalaman pohon
        'class_weight': ['balanced', None]   # Penting untuk Fraud (Imbalance handling)
    }

    # 4. Mulai MLflow Run untuk Tuning
    with mlflow.start_run(run_name="RandomForest_Manual_Log_Hyperparameter_Tuning"):
        print("Mencari parameter terbaik dengan GridSearchCV...")
        
        # Inisialisasi Model Dasar
        rf = RandomForestClassifier(random_state=42)
        
        # Inisialisasi Grid Search
        # cv=3 artinya melakukan 3-fold cross validation
        # scoring='f1_weighted' dipilih karena data fraud biasanya imbalance
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                   cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1)
        
        # Lakukan Training berulang-ulang (Fitting)
        grid_search.fit(X_train, y_train)

        # Ambil Hasil Terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Parameter Terbaik Ditemukan: {best_params}")
        print(f"Skor Validasi Terbaik (F1-Weighted): {best_score:.4f}")

        # --- LOGGING KE DAGSHUB (MANUAL) ---
        
        # A. Log Parameter Terbaik
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_method", "GridSearchCV")
        
        # B. Evaluasi Model Terbaik pada Data Test (Final Evaluation)
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # C. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # D. Log Artifacts (Visualisasi Model Terbaik)
        
        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens') # Pakai warna beda biar fresh
        plt.title(f'Best Model Confusion Matrix (F1: {f1:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_filename = "best_model_cm.png"
        plt.savefig(plot_filename)
        plt.close()
        mlflow.log_artifact(plot_filename)

        # Artifact 2: Classification Report
        report = classification_report(y_test, y_pred)
        report_filename = "best_model_report.txt"
        with open(report_filename, "w") as f:
            f.write(f"Hyperparameters: {best_params}\n\n") # Tulis params di dalam report text
            f.write(report)
        mlflow.log_artifact(report_filename)

        # E. Log Model Terbaik
        print("Menyimpan model terbaik ke DagsHub...")
        mlflow.sklearn.log_model(best_model, "best_model")

        # Cleanup
        if os.path.exists(plot_filename): os.remove(plot_filename)
        if os.path.exists(report_filename): os.remove(report_filename)

    print("=== Tuning Selesai. Cek DagsHub untuk melihat parameter pemenang. ===")

if __name__ == "__main__":
    run_tuning()