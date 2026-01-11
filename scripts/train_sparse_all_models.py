# train_sparse_all_models.py
"""
Extended benchmarking script (PLS-DA, XGB, RF, SVM).
Not used by the GUI; for reproducing confusion matrices & metrics in the paper.
"""

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.",
    category=UserWarning,
    module="sklearn.metrics._classification"
)

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    r2_score, mean_absolute_error,
    accuracy_score, precision_recall_fscore_support
)

from utils import build_sparse_features

RNG_SEED = 42
CSV = r"training_data/spectras.csv"


def save_confusion_matrix(cm, labels, title, fname, cmap):
    """Save full confusion matrix heatmap."""
    n = len(labels)
    size = max(6, 0.5 * n)

    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set(
        title=title,
        xlabel='Predicted',
        ylabel='True'
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_per_class(df, model_name):
    for metric in ['precision', 'recall', 'f1']:
        plt.figure(figsize=(16, 6))
        sns.barplot(x=df.index, y=df[metric])
        plt.xticks(rotation=90)
        plt.title(f"{model_name} per-class {metric}")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"{model_name.lower()}_{metric}.png")
        plt.close()


def build_sparse_train_test():
    """
    EXACTLY the same split logic as train_sparse.py:
    - 4 spectra per compound
    - train: replicates 0 and 2
    - test: replicate 1 (idx[1]; idx[3] ignored)
    Feature space: sparse peak intensities.
    """
    raw = pd.read_csv(CSV, header=None, delimiter=';', dtype=str)

    # wavenumber grid
    wav = raw.iloc[2:, 0].str.replace(',', '.').astype(float).values

    # all spectra (rows = wavenumbers, cols = spectra)
    spec_mat = raw.iloc[2:, 1:].map(lambda x: float(x.replace(',', '.')))

    # header with compound names
    header = [h.strip() for h in raw.iloc[0, 1:]]
    hdr_df = pd.DataFrame({'compound': header, 'spec_index': range(len(header))})

    train_feats, train_labels = [], []
    test_feats,  test_labels  = [], []

    for cmpd, grp in hdr_df.groupby('compound'):
        # must have 4 replicates
        if len(grp) != 4:
            continue

        idx = grp['spec_index'].tolist()
        train_idx = [idx[0], idx[2]]
        test_idx  = [idx[1], idx[3]]

        # training replicates
        for i in train_idx:
            abs_raw = spec_mat.iloc[:, i].values
            feat = build_sparse_features(abs_raw, wav)   # <── sparse peak features
            train_feats.append(feat)
            train_labels.append(cmpd)

        # ONE test replicate as in train_sparse.py (test_idx[0])
        i = test_idx[0]
        abs_raw = spec_mat.iloc[:, i].values
        feat = build_sparse_features(abs_raw, wav)
        test_feats.append(feat)
        test_labels.append(cmpd)

    X_train = pd.DataFrame(train_feats).fillna(0)
    y_train = pd.Series(train_labels, name='Label')
    X_test  = pd.DataFrame(test_feats).fillna(0).reindex(columns=X_train.columns, fill_value=0)
    y_test  = pd.Series(test_labels, name='Label')

    return wav, X_train, y_train, X_test, y_test


def train_sparse_models():
    # 1) Build sparse feature sets with *same* split as train_sparse.py
    wav, X_train, y_train, X_test, y_test = build_sparse_train_test()

    # 2) PCA (99% variance) – identical logic
    var_cum = np.cumsum(PCA().fit(X_train).explained_variance_ratio_)
    n_comp  = np.searchsorted(var_cum, 0.99) + 1
    print(f"PCA components used: {n_comp}")

    pca = PCA(n_components=n_comp, svd_solver='full', random_state=RNG_SEED)
    t0 = time.time()
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    print(f"PCA fit+transform: {time.time() - t0:.2f}s")

    # 3) PLS-DA (for completeness + PLS sparse model)
    lb  = LabelBinarizer().fit(y_train)
    Ytr = lb.transform(y_train)
    Yte = lb.transform(y_test)

    pls = PLSRegression(n_components=n_comp, tol=1e-6, max_iter=500)
    t0 = time.time()
    pls.fit(Xtr_pca, Ytr)
    print(f"PLS-DA train time: {time.time() - t0:.2f}s")

    Y_pred_pls = pls.predict(Xte_pca)
    y_pred_pls = lb.classes_[np.argmax(Y_pred_pls, axis=1)]

    print(f"PLS-DA R²:  {r2_score(Yte, Y_pred_pls, multioutput='uniform_average'):.3f}")
    print(f"PLS-DA MAE: {mean_absolute_error(Yte, Y_pred_pls, multioutput='uniform_average'):.3f}")

    acc_pls = accuracy_score(y_test, y_pred_pls)
    print(f"PLS-DA misclassification rate: {1 - acc_pls:.3f}")

    print("\n--- PLS-DA Classification Report ---")
    print(classification_report(y_test, y_pred_pls, zero_division=0))

    cm_pls = confusion_matrix(y_test, y_pred_pls)
    save_confusion_matrix(
        cm_pls, lb.classes_,
        'Confusion Matrix — PLS-DA (Sparse)',
        'confusion_pls_da_sparse.png', 'Blues'
    )

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_pls, zero_division=0)
    df_pls = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_pls, 'PLS-DA Sparse')

    # 4) XGBoost (same style as train_sparse.py, but we keep LabelEncoder)
    le = LabelEncoder().fit(y_train)
    xgb = XGBClassifier(
        eval_metric='mlogloss',
        subsample=0.7,
        random_state=RNG_SEED
    )
    t0 = time.time()
    xgb.fit(Xtr_pca, le.transform(y_train))
    print(f"XGB train time: {time.time() - t0:.2f}s")

    y_pred_xgb = le.inverse_transform(xgb.predict(Xte_pca))
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGB misclassification rate: {1 - acc_xgb:.3f}")

    print("\n--- XGBoost Classification Report ---")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    save_confusion_matrix(
        cm_xgb, le.classes_,
        'Confusion Matrix — XGBoost (Sparse)',
        'confusion_xgb_sparse.png', 'Oranges'
    )

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_xgb, zero_division=0)
    df_xgb = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=le.classes_)
    plot_per_class(df_xgb, 'XGBoost Sparse')

    # 5) Random Forest (same as train_sparse.py)
    rf = RandomForestClassifier(
        n_estimators=n_comp,
        random_state=RNG_SEED
    )
    t0 = time.time()
    rf.fit(Xtr_pca, y_train)
    print(f"RF train time: {time.time() - t0:.2f}s")

    y_pred_rf = rf.predict(Xte_pca)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"RF misclassification rate: {1 - acc_rf:.3f}")

    print("\n--- RF Classification Report ---")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    save_confusion_matrix(
        cm_rf, lb.classes_,
        'Confusion Matrix — RF (Sparse)',
        'confusion_rf_sparse.png', 'Greens'
    )

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, zero_division=0)
    df_rf = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_rf, 'RF Sparse')

    # 6) SVM (same as train_sparse.py)
    svc = SVC(probability=True, random_state=RNG_SEED)
    t0 = time.time()
    svc.fit(Xtr_pca, y_train)
    print(f"SVM train time: {time.time() - t0:.2f}s")

    y_pred_svc = svc.predict(Xte_pca)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print(f"SVM misclassification rate: {1 - acc_svc:.3f}")

    print("\n--- SVM Classification Report ---")
    print(classification_report(y_test, y_pred_svc, zero_division=0))

    cm_svc = confusion_matrix(y_test, y_pred_svc)
    save_confusion_matrix(
        cm_svc, lb.classes_,
        'Confusion Matrix — SVM (Sparse)',
        'confusion_svm_sparse.png', 'Purples'
    )

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_svc, zero_division=0)
    df_svc = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_svc, 'SVM Sparse')

    # 7) Save master pipeline (like train_sparse.py)
    columns = X_train.columns.tolist()
    pipeline = {
        'wav':     wav,
        'columns': columns,
        'pca':     pca,
        'pls':     pls,
        'xgb':     xgb,
        'rf':      rf,
        'svc':     svc,
        'lb':      lb,   # for PLS-DA
        'le_xgb':  le    # for XGB (integer labels)
    }
    joblib.dump(pipeline, 'ftir_model_pipeline.pkl')
    print("✅ Sparse training complete, master pipeline saved to ftir_model_pipeline.pkl")

    pipe_pls = {
        'wav': wav,
        'columns': columns,
        'pca': pca,
        'model': pls,
        'label_binarizer': lb
    }
    joblib.dump(pipe_pls, 'ftir_model_sparse_pls.pkl')
    print("✔ Saved ftir_model_sparse_pls.pkl")

    pipe_xgb = {
        'wav': wav,
        'columns': columns,
        'pca': pca,
        'model': xgb,
        'label_encoder': le
    }
    joblib.dump(pipe_xgb, 'ftir_model_sparse_xgb.pkl')
    print("✔ Saved ftir_model_sparse_xgb.pkl")

    pipe_rf = {
        'wav': wav,
        'columns': columns,
        'pca': pca,
        'model': rf
    }
    joblib.dump(pipe_rf, 'ftir_model_sparse_rf.pkl')
    print("✔ Saved ftir_model_sparse_rf.pkl")

    pipe_svm = {
        'wav': wav,
        'columns': columns,
        'pca': pca,
        'model': svc
    }
    joblib.dump(pipe_svm, 'ftir_model_sparse_svm.pkl')
    print("✔ Saved ftir_model_sparse_svm.pkl")


if __name__ == '__main__':
    train_sparse_models()
