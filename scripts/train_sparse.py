# train_sparse.py

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

from utils import build_sparse_features


def train_sparse():
    # ─── 1) Load & parse raw CSV ─────────────────────────────────────
    CSV = r"training_data/spectras.csv"
    raw = pd.read_csv(CSV, header=None, delimiter=';', dtype=str)

    # Wavenumber axis
    wav = (raw.iloc[2:, 0]
             .str.replace(',', '.')
             .astype(float)
             .values)

    # Spectral matrix (each column = one spectrum)
    spec_mat = raw.iloc[2:, 1:].map(lambda x: float(x.replace(',', '.')))

    # Header row: compound names for each spectrum
    header = [h.strip() for h in raw.iloc[0, 1:]]
    hdr_df = pd.DataFrame({'compound': header, 'spec_index': range(len(header))})

    # ─── 2) Build sparse train/test feature sets ─────────────────────
    train_feats, train_labels = [], []
    test_feats,  test_labels  = [], []

    for cmpd, grp in hdr_df.groupby('compound'):
        # only use compounds with all 4 replicates
        if len(grp) != 4:
            continue

        idx = grp['spec_index'].tolist()
        t_idx = [idx[0], idx[2]]   # training replicates
        e_idx = [idx[1], idx[3]]   # test replicate(s)

        # training samples
        for i in t_idx:
            abs_raw = spec_mat.iloc[:, i].values
            feat = build_sparse_features(abs_raw, wav)  # dict of Peak_*: intensity
            train_feats.append(feat)
            train_labels.append(cmpd)

        # one test sample per compound
        abs_raw = spec_mat.iloc[:, e_idx[0]].values
        feat = build_sparse_features(abs_raw, wav)
        test_feats.append(feat)
        test_labels.append(cmpd)

    # ─── 3) Convert to DataFrames / arrays ───────────────────────────
    X_train = pd.DataFrame(train_feats).fillna(0)
    X_test  = (pd.DataFrame(test_feats)
                 .fillna(0)
                 .reindex(columns=X_train.columns, fill_value=0))

    y_train = np.array(train_labels)
    y_test  = np.array(test_labels)

    print(f"[SPARSE] Feature matrix shape: {X_train.shape}")

    # ─── 4) Label binarizer for PLS-DA ───────────────────────────────
    lb = LabelBinarizer().fit(y_train)
    Ytr = lb.transform(y_train)

    # ─── 5) PCA on sparse features (99% variance) ────────────────────
    var_cum = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_comp = np.searchsorted(var_cum, 0.99) + 1
    print(f"[SPARSE] PCA components (99% var): {n_comp}")

    pca = PCA(
        n_components=n_comp,
        svd_solver='full',
        random_state=42
    )

    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)

    # ─── 6) PLS-DA on sparse PCA scores ──────────────────────────────
    pls = PLSRegression(
        n_components=n_comp,
        max_iter=2000
    )
    pls.fit(Xtr_p, Ytr)

    # ─── 7) Evaluate on held-out sparse test set ─────────────────────
    Yte = lb.transform(y_test)
    Y_pred = pls.predict(Xte_p)
    y_pred = lb.classes_[np.argmax(Y_pred, axis=1)]

    print("\n=== SPARSE PLS-DA on held-out replicates ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # ─── 8) Save sparse pipeline ─────────────────────────────────────
    pipeline = {
        'pca': pca,
        'pls': pls,
        'lb': lb,
        'wav': wav,
        'columns': X_train.columns.tolist(),
        'feature_type': 'sparse'
    }

    joblib.dump(pipeline, "ftir_model_sparse.pkl")
    print("✔ SPARSE model saved as ftir_model_sparse.pkl")


if __name__ == "__main__":
    train_sparse()
