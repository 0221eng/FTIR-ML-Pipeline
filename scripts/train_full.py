# train_full.py

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

from utils import build_full_features


def train_full():
    # ─── 1) Load & parse raw CSV ─────────────────────────────────────
    CSV = r"training_data/spectras.csv"
    raw = pd.read_csv(CSV, header=None, delimiter=';', dtype=str)

    # Wavenumber axis (full range; cropping happens inside build_full_features)
    wav = (raw.iloc[2:, 0]
             .str.replace(',', '.')
             .astype(float)
             .values)

    # Spectral matrix (each column = one spectrum)
    spec_mat = raw.iloc[2:, 1:].map(lambda x: float(x.replace(',', '.')))

    # Header row: compound names for each spectrum
    header = [h.strip() for h in raw.iloc[0, 1:]]
    hdr_df = pd.DataFrame({'compound': header, 'spec_index': range(len(header))})

    # ─── 2) Build full-resolution train/test sets ────────────────────
    train_X, train_y = [], []
    test_X,  test_y  = [], []

    for cmpd, grp in hdr_df.groupby('compound'):
        if len(grp) != 4:
            continue

        idx = grp['spec_index'].tolist()
        t_idx = [idx[0], idx[2]]   # training replicates
        e_idx = [idx[1], idx[3]]   # test replicates

        # training samples
        for i in t_idx:
            abs_raw = spec_mat.iloc[:, i].values
            feat = build_full_features(abs_raw, wav)   # 1D array (500–3000 cm⁻¹, normalized)
            train_X.append(feat)
            train_y.append(cmpd)

        # one test sample per compound
        abs_raw = spec_mat.iloc[:, e_idx[0]].values
        feat = build_full_features(abs_raw, wav)
        test_X.append(feat)
        test_y.append(cmpd)

    X_train = np.vstack(train_X)
    X_test  = np.vstack(test_X)
    y_train = np.array(train_y)
    y_test  = np.array(test_y)

    print(f"[FULL] Feature matrix shape: {X_train.shape}")

    # ─── 3) PCA on full features (99% variance) ──────────────────────
    var_cum = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_comp = np.searchsorted(var_cum, 0.99) + 1
    print(f"[FULL] PCA components (99% var): {n_comp}")

    pca = PCA(
        n_components=n_comp,
        svd_solver='full',
        random_state=42
    )

    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)

    # ─── 4) PLS-DA on PCA scores ─────────────────────────────────────
    lb = LabelBinarizer().fit(y_train)
    Ytr = lb.transform(y_train)

    pls = PLSRegression(
        n_components=n_comp,
        max_iter=2000
    )
    pls.fit(Xtr_p, Ytr)

    # ─── 5) Evaluate on held-out test set ────────────────────────────
    Yte = lb.transform(y_test)
    Y_pred = pls.predict(Xte_p)
    y_pred = lb.classes_[np.argmax(Y_pred, axis=1)]

    print("\n=== FULL-spectrum PLS-DA on held-out replicates ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # ─── 6) Save full pipeline ───────────────────────────────────────
    pipeline = {
        'pca': pca,
        'pls': pls,
        'lb':  lb,
        'wav': wav,
        'feature_type': 'full'
    }

    joblib.dump(pipeline, "ftir_model_full.pkl")
    print("✔ FULL-spectrum model saved as ftir_model_full.pkl")


if __name__ == "__main__":
    train_full()
