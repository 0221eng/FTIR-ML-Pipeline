# utils.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

# ── Hyperparameters ──────────────────────────────────────────────────────────
WINDOW_LENGTH        = 151
POLY_ORDER           = 3
LAM                  = 1e8
P_PARAM              = 0.01
N_ITER               = 10
PERCENTILE_THRESHOLD = 65   # percentile for peak selection
PEAK_TOLERANCE       = 10   # cm⁻¹ tolerance around peak position
BIN_WIDTH            = 4    # cm⁻¹, for peak binning in sparse features


# ── Baseline correction (ALS) ───────────────────────────────────────────────
def als_baseline(y, lam=LAM, p=P_PARAM, niter=N_ITER):
    """Asymmetric least squares baseline removal."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + D
        Z = csc_matrix(Z)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


# ── Common preprocessing core ───────────────────────────────────────────────
def preprocess_spectrum(abs_raw, wav,
                        window_length=WINDOW_LENGTH,
                        poly_order=POLY_ORDER,
                        lam=LAM,
                        p=P_PARAM,
                        niter=N_ITER):
    """
    SG smoothing + ALS baseline + area normalization.
    Returns (wav_region, norm_region) restricted to 500–3000 cm⁻¹.
    This is the common core used by both full and sparse representations.
    """
    # 1) Savitzky–Golay smoothing
    y_sm = savgol_filter(abs_raw, window_length, poly_order)

    # 2) Baseline (ALS)
    bl = als_baseline(y_sm, lam=lam, p=p, niter=niter)

    # 3) Baseline-corrected + area normalization
    diff = y_sm - bl
    area = np.trapezoid(diff, wav)
    if area < 0:
        area *= -1
    norm = diff / area if area != 0 else diff

    # Restrict to 500–3000 cm⁻¹
    mask = (wav >= 500) & (wav <= 3000)
    return wav[mask], norm[mask]


# ── Representation (i): full-resolution features ────────────────────────────
def build_full_features(abs_raw, wav):
    """
    Representation (i): full-resolution preprocessed spectrum
    in the 500–3000 cm⁻¹ region.

    Returns
    -------
    1D numpy array of normalized absorbance values.
    """
    _, norm_region = preprocess_spectrum(abs_raw, wav)
    return norm_region


# ── Representation (ii): peak-sparse features ───────────────────────────────
def build_sparse_features(abs_raw, wav,
                          percentile=PERCENTILE_THRESHOLD,
                          tol=PEAK_TOLERANCE,
                          bin_width=BIN_WIDTH):
    """
    Representation (ii): peak-sparse feature set using
    percentile-based peak selection + wavenumber binning.

    Returns
    -------
    dict: { "Peak_<bin>": intensity }
          where <bin> is the binned wavenumber (e.g. 1000, 1004, ...).
    """
    wav_region, norm_region = preprocess_spectrum(abs_raw, wav)

    df = pd.DataFrame({'wavenumbers': wav_region, 'absorbance': norm_region})
    thr = np.percentile(df['absorbance'], percentile)
    peaks = df[df['absorbance'] >= thr].copy()

    feats = {}
    for pk in peaks['wavenumbers']:
        # Bin wavenumber to a fixed grid (e.g. 4 cm⁻¹)
        pk_bin = int(round(pk / bin_width) * bin_width)
        idx = np.argmin(np.abs(wav_region - pk))
        if abs(wav_region[idx] - pk) <= tol:
            key = f"Peak_{pk_bin}"
            val = norm_region[idx]
            # If multiple peaks land in the same bin, keep the max intensity
            if key in feats:
                feats[key] = max(feats[key], val)
            else:
                feats[key] = val
    return feats


# LEGACY, only used for old experiments – not part of the main pipeline

def spectrum_to_features(abs_raw, wav):
    """
    Smooth + remove baseline + normalize + pick top peaks.
    Returns DataFrame of peaks and the normalized FULL spectrum (no mask),
    for compatibility with existing train_sparse.py.
    """
    # 1) Savitzky–Golay smoothing
    y_sm = savgol_filter(abs_raw, WINDOW_LENGTH, POLY_ORDER)
    # 2) Baseline
    bl = als_baseline(y_sm)
    # 3) Area normalization
    diff = y_sm - bl
    area = np.trapezoid(diff, wav)
    if area < 0:
        area *= -1
    norm = diff / area if area != 0 else diff

    # 4) build DataFrame & limit region only for peak selection
    df = pd.DataFrame({'wavenumbers': wav, 'absorbance': norm})
    df = df[(df['wavenumbers'] >= 500) & (df['wavenumbers'] <= 3000)]

    # 5) select top percentile peaks
    thr = np.percentile(df['absorbance'], PERCENTILE_THRESHOLD)
    peaks = df[df['absorbance'] >= thr].copy()
    peaks['Average_Wavenumber'] = peaks['wavenumbers']

    return peaks, norm


def extract_peak_intensities(wav, norm, peaks, tol=PEAK_TOLERANCE):
    """
    Given a normalized spectrum (norm) and a peaks DataFrame (from
    spectrum_to_features), return a dict of
        Peak_<wavenumber>: intensity
    for each matched peak.

    This is kept in the original style for compatibility with train_sparse.py.
    """
    feats = {}
    for pk in peaks['Average_Wavenumber']:
        idx = np.argmin(np.abs(wav - pk))
        if abs(wav[idx] - pk) <= tol:
            feats[f'Peak_{pk:.2f}'] = norm[idx]
        else:
            feats[f'Peak_{pk:.2f}'] = 0
    return feats
