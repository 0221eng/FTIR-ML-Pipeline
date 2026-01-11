# benchmark_gui_sparse.py

import time
import joblib
import pandas as pd
import numpy as np

from utils import build_sparse_features


def benchmark_gui_like_sparse(csv_path, n_repeats=100):
    """
    Benchmark GUI-like latency for the SPARSE PLS-DA model.

    It mimics what the GUI does:
      - take a raw spectrum (absorbance vs wavenumber)
      - apply the SAME sparse feature extraction as train_sparse.py
      - PCA transform
      - PLS-DA predict
    """
    # Load pipeline ONCE (like app startup)
    pipe = joblib.load("ftir_model_sparse.pkl")
    pca   = pipe["pca"]
    pls   = pipe["pls"]
    lb    = pipe["lb"]
    wav   = pipe["wav"]
    cols  = pipe["columns"]

    # Load raw spectrum ONCE (like GUI reading from spectrometer)
    df = pd.read_csv(csv_path, header=None, delimiter=";")
    abs_raw = (
        df.iloc[2:, 1]
          .astype(str)
          .str.replace(",", ".")
          .astype(float)
          .values
    )

    times_ms = []
    last_pred = None

    for _ in range(n_repeats):
        t0 = time.perf_counter()

        feats = build_sparse_features(abs_raw, wav)

        # align to training feature space
        X_unk = (
            pd.DataFrame([feats])
              .reindex(columns=cols, fill_value=0)
              .values
        )

        # PCA + PLS-DA
        Xp    = pca.transform(X_unk)
        probs = pls.predict(Xp).ravel()
        last_pred = lb.classes_[np.argmax(probs)]

        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = np.mean(times_ms)
    max_ms  = np.max(times_ms)

    print(f"\n🔬 Last predicted compound: {last_pred}")
    print(f"GUI-like SPARSE benchmark over {n_repeats} runs:")
    print(f"  Mean: {mean_ms:.2f} ms   Max: {max_ms:.2f} ms")


if __name__ == "__main__":
    # You can later change this to e.g. "examples/acetaldehyde.csv"
    benchmark_gui_like_sparse("acetaldehyde.csv", n_repeats=100)
