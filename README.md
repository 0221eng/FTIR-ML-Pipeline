# FTIR-ML-Pipeline

FTIR-ML-Pipeline is a Python workflow for **classification of FTIR spectra** using
PLS-DA and other ML models, plus a **desktop GUI** for single-spectrum prediction.
The core idea is to compare a full-resolution representation of the spectrum with a
sparse, peak-based representation.

---

## Intellectual Property / Registration

The work described in this manuscript is covered by a patent application currently under
review by the Intellectual Property Agency of the Republic of Uzbekistan (Ref: DT
202509952).


---

## Large Files (Google Drive)

For convenience all the contents of the repository are available in Google Drive as well.
They are available in a shared Google Drive folder, including:

- `training_data/spectras.csv` – full FTIR training dataset (4 replicates per compound)
- Example spectra for GUI testing (CSV files)
- Pre-trained model files (`*.pkl`), if you choose to share them
- Packaged GUI executable (e.g. `FTIRApp.zip`), if applicable
- A copy of the Uzbek registration / patent document

👉 **Google Drive folder** (data, models, app, docs):  
- [`FTIR_ML_Pipeline`](https://drive.google.com/drive/folders/1b0MHdIZJXpZ08Sr2WzCE9Pl5s9wH96Xw?usp=drive_link)

---

## Features

- Unified preprocessing (Savitzky–Golay, ALS baseline correction, area normalisation)
- Two feature types:
  - Full spectrum (500–3000 cm⁻¹)
  - Sparse peak features (percentile-based peak selection + binning)
- Training scripts for:
  - Full-resolution PLS-DA (`train_full.py`)
  - Sparse PLS-DA (`train_sparse.py`)
  - Extended benchmark with PLS-DA, XGBoost, RF, SVM (`train_sparse_all_models.py`)
- Benchmark scripts:
  - GUI-like latency for sparse model (`benchmark_gui_sparse.py`)
  - Full vs sparse inference speed & model size (`benchmark_two_models.py`)
- PyQt5 GUI (`app.py`) to load a CSV spectrum, plot it, run prediction and export a PDF report
- Reusable preprocessing utilities in `utils.py`

---

## Repository structure

```text
FTIR-ML-Pipeline/
├─ app.py                     # PyQt5 GUI for single-spectrum prediction
├─ utils.py                   # Shared preprocessing & feature-extraction functions
│
├─ data/                      # Small example spectra for GUI testing (CSV)
│    acetaldehyde.csv
│    hydrazine.csv
│    ...
│
├─ training_data/
│    spectras.csv             # Full FTIR training dataset (4 replicates / compound)
│                             # (large file – usually downloaded from Google Drive)
│
├─ models/                    # Optional: saved model weights / utilities
│    models.py
│    PLS_DA.py
│    PLS_DA_full.py
│    RF_sparse.py
│    SVM_sparse.py
│    XGB_sparse.py
│
└─ scripts/
     benchmark_gui_sparse.py      # GUI-style latency benchmark (sparse PLS-DA)
     benchmark_two_models.py      # Full vs sparse speed / feature count / size
     train_full.py                # Full-spectrum PLS-DA
     train_sparse.py              # Sparse PLS-DA
     train_sparse_all_models.py   # Sparse PLS-DA, XGB, RF, SVM + confusion matrices
```

## Authors

 - Babaa Moulay Rachid (correspodning author; mail to: [`m.babaa@newuu.uz `](m.babaa@newuu.uz ))

 - Atabaev Otabek

 - Samandarov Shakhzodbek
