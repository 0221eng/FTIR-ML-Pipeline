# XGB_sparse.py
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import build_sparse_features  # unified sparse featurizer


def classify(csv_path, n_repeats=50):
    """
    Sparse-peak XGBoost inference with latency benchmarking.

    Parameters
    ----------
    csv_path : str
        Path to a single-spectrum CSV (same format as training file).
    n_repeats : int
        Number of repeated runs for latency measurement
        (model is loaded only once).
    """
    # 0) Load pipeline ONCE (like a real app startup)
    pipe = joblib.load("ftir_model_sparse_xgb.pkl")
    pca   = pipe["pca"]
    model = pipe["model"]
    wav   = pipe["wav"]
    cols  = pipe["columns"]
    le    = pipe["label_encoder"]  # LabelEncoder used during training

    total_ms = []
    last_probs = None
    last_pred = None

    for _ in range(n_repeats):
        t0 = time.perf_counter()

        # 1) Load a single spectrum from CSV
        df = pd.read_csv(csv_path, header=None, delimiter=";")
        abs_raw = (
            df.iloc[2:, 1]
              .astype(str)
              .str.replace(",", ".")
              .astype(float)
              .values
        )

        # 2) Sparse feature vector (same as training)
        feat_dict = build_sparse_features(abs_raw, wav)
        X_unk = (
            pd.DataFrame([feat_dict])
              .reindex(columns=cols, fill_value=0)
        )

        # 3) PCA + XGBoost prediction (probabilities)
        Xp = pca.transform(X_unk)
        probs = model.predict_proba(Xp)[0]  # shape (n_classes,)
        idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([idx])[0]

        t1 = time.perf_counter()
        total_ms.append((t1 - t0) * 1000.0)  # ms

        last_probs = probs
        last_pred  = pred_label

    # Reporting
    avg_ms = np.mean(total_ms)
    max_ms = np.max(total_ms)

    classes = le.classes_

    print(f"\n🔬 Predicted compound (XGBoost, sparse): {last_pred}\n")
    print("Class probabilities:")
    for cls, p in zip(classes, last_probs):
        print(f"  {cls:30s} {p:.4f}")

    print(f"\n⏱ End-to-end latency over {n_repeats} run(s):")
    print(f"  Mean: {avg_ms:.2f} ms   Max: {max_ms:.2f} ms")

    # Plot (NOT timed)
    plt.bar(classes, last_probs)
    plt.xticks(rotation=90)
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities (XGBoost, sparse features)")
    plt.tight_layout()
    plt.show()

    return avg_ms, max_ms
# any test file can be put

classify("acetaldehyde.csv")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python XGB_sparse.py <path_to_spectrum.csv>")
        sys.exit(1)
    classify(sys.argv[1])
