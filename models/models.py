# train.py
import time
import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def save_confusion_matrix(cm, labels, title, fname, cmap):
    """
    Save the full confusion matrix `cm` as three heatmap images:
    classes 1–30, 31–60, and 61–end (e.g. 89).

    Parameters
    ----------
    cm : array‐like, shape (n_classes, n_classes)
        The confusion matrix to plot.
    labels : list of str, length n_classes
        The class labels for the axes.
    title : str
        The title to put on the figure.
    fname : str
        The base filename (including path) where the images will be saved.
    cmap : matplotlib colormap
        The colormap to use for the heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    n = len(labels)

    # If there are 30 or fewer classes, just do a single figure as before
    if n <= 30:
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
        return

    # Otherwise, split into three parts: 1–30, 31–60, 61–end
    ranges = [(0, 30), (30, 60), (60, n)]

    for idx, (start, end) in enumerate(ranges, start=1):
        if start >= n:
            break  # in case n < upper bound of last range

        cm_part = np.array(cm)[start:end, start:end]
        labels_part = labels[start:end]

        n_part = len(labels_part)
        size = max(6, 0.5 * n_part)

        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            cm_part,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=labels_part,
            yticklabels=labels_part,
            ax=ax
        )
        ax.set(
            title=f"{title} (classes {start+1}–{end})",
            xlabel='Predicted',
            ylabel='True'
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()
        fig.savefig(f"{fname}_part{idx}.png")
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


def train():
    # ─── 1) Load & parse raw CSV ─────────────────────────────────────
    CSV = r"training_data/spectras.csv"
    raw = pd.read_csv(CSV, header=None, delimiter=';', dtype=str)
    wav = raw.iloc[2:, 0].str.replace(',', '.').astype(float).values
    spec_mat = raw.iloc[2:, 1:].map(lambda x: float(x.replace(',', '.')))
    header = [h.strip() for h in raw.iloc[0, 1:]]
    hdr_df = pd.DataFrame({'compound': header, 'spec_index': range(len(header))})

    # ─── 2) Build train/test feature sets (sparse peak features) ─────
    train_feats, train_labels = [], []
    test_feats, test_labels = [], []
    for cmpd, grp in hdr_df.groupby('compound'):
        if len(grp) != 4:
            continue
        idx = grp['spec_index'].tolist()
        train_idx, test_idx = [idx[0], idx[2]], [idx[1], idx[3]]

        # two training replicates per compound
        for i in train_idx:
            abs_raw = spec_mat.iloc[:, i].values
            feat = build_sparse_features(abs_raw, wav)  # <── HERE
            train_feats.append(feat)
            train_labels.append(cmpd)

        # one test replicate
        abs_raw = spec_mat.iloc[:, test_idx[0]].values
        feat = build_sparse_features(abs_raw, wav)  # <── AND HERE
        test_feats.append(feat)
        test_labels.append(cmpd)

    X_train = pd.DataFrame(train_feats).fillna(0)
    y_train = pd.Series(train_labels, name='Label')
    X_test  = pd.DataFrame(test_feats).fillna(0).reindex(columns=X_train.columns, fill_value=0)
    y_test  = pd.Series(test_labels,  name='Label')

    # ─── 3) PCA → PLS-DA setup ────────────────────────────────────────
    lb  = LabelBinarizer().fit(y_train)
    Ytr = lb.transform(y_train)
    Yte = lb.transform(y_test)

    # ─── 3) PCA → choose #components by cumulative variance ─────────────
    pca_full = PCA()
    pca_full.fit(X_train)
    var_cum = np.cumsum(pca_full.explained_variance_ratio_)

    chosen_var = 0.99
    n_comp     = np.searchsorted(var_cum, chosen_var) + 1
    print(f"PCA components used: {n_comp}")

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_cum) + 1), var_cum)
    plt.axhline(y=chosen_var, linestyle='--')           # horizontal line at chosen variance
    plt.axvline(x=n_comp, linestyle='--')               # vertical line at chosen n_components
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA cumulative explained variance (sparse features)')
    plt.tight_layout()
    plt.savefig('pca_cumulative_variance_sparse.png')
    plt.close()

    # Now fit PCA with the chosen number of components
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=42)
    start = time.time()
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    print(f"PCA fit+transform: {time.time() - start:.2f}s")


    # ─── 4) PLS-DA ─────────────────────────────────────────────────────
    pls = PLSRegression(n_components=n_comp, tol=1e-6, max_iter=500)
    start = time.time()
    pls.fit(Xtr_pca, Ytr)
    print(f"PLS-DA train time: {time.time() - start:.2f}s")

    Y_pred_pls = pls.predict(Xte_pca)
    y_pred_pls = lb.classes_[np.argmax(Y_pred_pls, axis=1)]

    # regression metrics
    print(f"PLS-DA R²:  {r2_score(Yte, Y_pred_pls, multioutput='uniform_average'):.3f}")
    print(f"PLS-DA MAE: {mean_absolute_error(Yte, Y_pred_pls, multioutput='uniform_average'):.3f}")

    acc_pls = accuracy_score(y_test, y_pred_pls)
    print(f"PLS-DA misclassification rate: {1 - acc_pls:.3f}")

    print("\n--- PLS-DA Classification Report ---")
    print(classification_report(y_test, y_pred_pls, zero_division=0))

    cm_pls = confusion_matrix(y_test, y_pred_pls)
    save_confusion_matrix(cm_pls, lb.classes_,
                           'Confusion Matrix — PLS-DA',
                           'confusion_pls_da', 'Blues')

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_pls, zero_division=0)
    df_pls = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_pls, 'PLS-DA')

    # ─── 5) XGBoost ─────────────────────────────────────────────────────
    le = LabelEncoder().fit(y_train)
    xgb = XGBClassifier(eval_metric='mlogloss', subsample=0.7, random_state=RNG_SEED)
    start = time.time()
    xgb.fit(Xtr_pca, le.transform(y_train))
    print(f"XGB train time: {time.time() - start:.2f}s")

    y_pred_xgb = le.inverse_transform(xgb.predict(Xte_pca))
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGB misclassification rate: {1 - acc_xgb:.3f}")

    print("\n--- XGBoost Classification Report ---")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    save_confusion_matrix(cm_xgb, le.classes_,
                           'Confusion Matrix — XGBoost',
                           'confusion_xgb', 'Oranges')

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_xgb, zero_division=0)
    df_xgb = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=le.classes_)
    plot_per_class(df_xgb, 'XGBoost')

    # ─── 6) Random Forest ────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=n_comp, random_state=RNG_SEED)
    start = time.time()
    rf.fit(Xtr_pca, y_train)
    print(f"RF train time: {time.time() - start:.2f}s")

    y_pred_rf = rf.predict(Xte_pca)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"RF misclassification rate: {1 - acc_rf:.3f}")

    print("\n--- RF Classification Report ---")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    save_confusion_matrix(cm_rf, lb.classes_,
                           'Confusion Matrix — RF',
                           'confusion_rf', 'Greens')

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_rf, zero_division=0)
    df_rf = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_rf, 'RF')

    # ─── 7) SVM ──────────────────────────────────────────────────────────
    svc = SVC(probability=True, random_state=RNG_SEED)
    start = time.time()
    svc.fit(Xtr_pca, y_train)
    print(f"SVM train time: {time.time() - start:.2f}s")

    y_pred_svc = svc.predict(Xte_pca)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print(f"SVM misclassification rate: {1 - acc_svc:.3f}")

    print("\n--- SVM Classification Report ---")
    print(classification_report(y_test, y_pred_svc, zero_division=0))

    cm_svc = confusion_matrix(y_test, y_pred_svc)
    save_confusion_matrix(cm_svc, lb.classes_,
                           'Confusion Matrix — SVM',
                           'confusion_svm', 'Purples')

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_svc, zero_division=0)
    df_svc = pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1}, index=lb.classes_)
    plot_per_class(df_svc, 'SVM')

    # ─── 8) Persist pipeline ─────────────────────────────────────────────
    pipeline = {
        'wav':     wav,
        'columns': X_train.columns.tolist(),
        'pca':     pca,
        'pls':     pls,
        'xgb':     xgb,
        'rf':      rf,
        'svc':     svc,
        'lb':      lb
    }
    joblib.dump(pipeline, 'ftir_model_pipeline.pkl')
    print("✅ Training complete, pipeline saved to ftir_model_pipeline.pkl")




if __name__ == '__main__':
    train()
