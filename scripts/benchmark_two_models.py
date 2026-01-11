# benchmark_two_models.py
import time
import joblib
import numpy as np
import pandas as pd
from utils import spectrum_to_features
import os
from utils import build_sparse_features
CSV = r"training_data/spectras.csv"
raw = pd.read_csv(CSV, header=None, delimiter=';', dtype=str)

wav = raw.iloc[2:,0].str.replace(',','.').astype(float).values
spec_mat = raw.iloc[2:,1:].map(lambda x: float(x.replace(',','.')))

# Load both models
full = joblib.load("ftir_model_full.pkl")
sparse = joblib.load("ftir_model_sparse.pkl")

# pick N spectra
N = 100
indices = np.random.choice(spec_mat.shape[1], size=N, replace=False)

# ---------- Full inference ----------
def infer_full(abs_raw):
    _, norm = spectrum_to_features(abs_raw, wav)
    df = pd.DataFrame({'wav':wav, 'norm':norm})
    df = df[(df['wav'] >= 500) & (df['wav'] <= 3000)]
    x = df['norm'].values.reshape(1, -1)
    x_p = full['pca'].transform(x)
    y = full['pls'].predict(x_p)
    return y

# ---------- Sparse inference ----------


def infer_sparse(abs_raw):
    feat = build_sparse_features(abs_raw, wav)   # same as train_sparse.py
    x = pd.DataFrame([feat]).reindex(columns=sparse['columns'], fill_value=0).values
    x_p = sparse['pca'].transform(x)
    y = sparse['pls'].predict(x_p)
    return y

# Benchmarking
t_full = []
t_sparse = []

for i in indices:
    abs_raw = spec_mat.iloc[:,i].values

    t0 = time.time()
    infer_full(abs_raw)
    t_full.append((time.time()-t0)*1000)

    t0 = time.time()
    infer_sparse(abs_raw)
    t_sparse.append((time.time()-t0)*1000)

# Outputs
print("\n=== BENCHMARK RESULTS ===")
print(f"Full resolution mean time:   {np.mean(t_full):.2f} ms")
print(f"Sparse feature mean time:    {np.mean(t_sparse):.2f} ms")
print(f"Speed-up: {100*(1 - np.mean(t_sparse)/np.mean(t_full)):.1f}%")

# Feature counts
F_full = len(wav[(wav >= 500) & (wav <= 3000)])
F_sparse = len(sparse["columns"])

print("\nFeature counts")
print("F_full   =", F_full)
print("F_sparse =", F_sparse)

# Model sizes
print("\nModel sizes")
print("Full model:   ", os.path.getsize("ftir_model_full.pkl")/1024, "KB")
print("Sparse model: ", os.path.getsize("ftir_model_sparse.pkl")/1024, "KB")
