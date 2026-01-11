# PLS_DA_full.py
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import spectrum_to_features  # full pipeline uses only this


def classify(csv_path, n_repeats=50):
    """
    Full-spectrum PLS-DA inference with latency benchmarking.

    Parameters
    ----------
    csv_path : str
        Path to a single-spectrum CSV (same format as training file).
    n_repeats : int
        Number of repeated runs for latency measurement.
    """
    # 0) Load pipeline ONCE
    pipe = joblib.load("ftir_model_full.pkl")
    pca, pls, lb = pipe["pca"], pipe["pls"], pipe["lb"]
    wav = pipe["wav"]

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

        # 2) Preprocess & build FULL feature vector
        #    (same logic as in train_full.py)
        _, norm = spectrum_to_features(abs_raw, wav)
        df_spec = pd.DataFrame({"wav": wav, "norm": norm})
        df_spec = df_spec[(df_spec["wav"] >= 500) & (df_spec["wav"] <= 3000)]
        x_vec = df_spec["norm"].values.reshape(1, -1)  # shape (1, n_points)

        # 3) PCA + PLS-DA prediction
        Xp = pca.transform(x_vec)
        probs = pls.predict(Xp).ravel()
        pred = lb.classes_[np.argmax(probs)]

        t1 = time.perf_counter()
        total_ms.append((t1 - t0) * 1000.0)

        last_probs = probs
        last_pred = pred

    # Reporting
    avg_ms = np.mean(total_ms)
    max_ms = np.max(total_ms)

    print(f"\n🔬 Predicted compound: {last_pred}\n")
    print("Prediction scores:")
    for cls, p in zip(lb.classes_, last_probs):
        print(f"  {cls:30s} {p:.4f}")

    print(f"\n⏱ End-to-end latency over {n_repeats} run(s):")
    print(f"  Mean: {avg_ms:.2f} ms   Max: {max_ms:.2f} ms")

    # Plot (NOT timed)
    plt.bar(lb.classes_, last_probs)
    plt.xticks(rotation=90)
    plt.ylabel("PLS-DA score")
    plt.title("Prediction Scores (Full spectrum)")
    plt.tight_layout()
    plt.show()

    return avg_ms, max_ms
# any test file can be put
classify("acetaldehyde.csv")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python PLS_DA_full.py <path_to_spectrum.csv>")
        sys.exit(1)
    classify(sys.argv[1])
