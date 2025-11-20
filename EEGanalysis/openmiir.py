#!/usr/bin/env python3
"""
openmiir_ei.py
Compute an Engagement Index (EI) per song/clip from OpenMIIR EEG.

Features:
  - Frontal alpha suppression (-alpha)
  - Beta power
  - Low-gamma power
  - Spectral entropy (4-45 Hz)
EI (per window) = z(-alpha_F) + z(beta_F) + z(gamma_F) + z(spectral_entropy_F)
EI (per clip)   = mean over windows within that clip

Assumptions:
  - Input files are MNE-readable FIF from OpenMIIR.
  - Clip onsets/durations are available via Annotations whose description contains a clip id
    like "Stimulus/clip_01" or "clip_01". Adapt the pattern in `_extract_clip_segments`.
  - Using listening trials only (filter by keyword if present in description).

Usage:
  python openmiir_ei.py --data_dir /path/to/openmiir/fif --subject S01 --out ai_clips.csv
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

# ---- Config ----
FRONTAL_CHANNELS = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"]
BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "lgam":  (30.0, 45.0),
}
PSD_FMIN, PSD_FMAX = 4.0, 45.0       # for spectral entropy
HP, LP = 1.0, 45.0                   # preprocessing band-pass
NOTCH = 50.0                         # adjust if recordings used 60 Hz mains
WIN_SEC, STEP_SEC = 2.0, 1.0         # windowing for features
RESAMPLE_HZ = 256                    # resample for speed and stability


# ---- Helpers ----
def _pick_frontal(raw: mne.io.BaseRaw):
    chs = [ch for ch in FRONTAL_CHANNELS if ch in raw.ch_names]
    if not chs:
        raise RuntimeError("No frontal channels found. Check channel names.")
    return mne.pick_channels(raw.ch_names, chs)


def _extract_clip_segments(raw: mne.io.BaseRaw):
    """
    Return dict: clip_id -> list of (onset, offset) in seconds.
    Adjust the patterns for your annotation scheme.
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        raise RuntimeError("No annotations present. You need clip onsets/durations.")

    # Pattern examples in OpenMIIR often include 'Stimulus/...' or 'clip_XX'
    pat = re.compile(r"(?:Stimulus/)?clip[_\- ]?(\d+)", re.IGNORECASE)

    segments = defaultdict(list)
    for ann in raw.annotations:
        desc = ann["description"]
        m = pat.search(desc)
        if not m:
            continue
        clip_id = f"clip_{int(m.group(1)):02d}"

        # Optional: keep only listening trials if annotated
        if re.search(r"imag", desc, re.IGNORECASE):
            continue  # skip imagination by default

        onset = float(ann["onset"])
        dur = float(ann["duration"]) if ann["duration"] is not None else 0.0
        if dur <= 0:
            # Fallback: if duration is missing, you may set a fixed length
            # or skip. Here we skip.
            continue
        segments[clip_id].append((onset, onset + dur))

    if not segments:
        raise RuntimeError("No clip segments matched. Adjust the regex/policy above.")
    return segments


def _preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw.load_data()
    # BioSemi 64 montage is typical for OpenMIIR; set if names match
    try:
        raw.set_montage("biosemi64", match_case=False, on_missing="ignore")
    except Exception:
        pass
    raw.notch_filter(NOTCH, picks="eeg", verbose=False)
    raw.filter(HP, LP, picks="eeg", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.resample(RESAMPLE_HZ)
    return raw


def _fixed_length_epochs(raw_segment: mne.io.BaseRaw, duration=WIN_SEC, step=STEP_SEC):
    return mne.make_fixed_length_epochs(
        raw_segment, duration=duration, overlap=(duration - step), preload=True
    )


def _bandpower(eeg: np.ndarray, sfreq: float, fmin: float, fmax: float):
    """
    eeg: (n_epochs, n_channels, n_times)
    Returns mean band power per epoch across channels.
    """
    n_ep, n_ch, n_t = eeg.shape
    # Welch PSD per epoch/channel
    # Use a 1 s window for PSD estimation inside each 2 s epoch
    nperseg = int(sfreq * 1.0)
    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1, average="mean")
    band_mask = (freqs >= fmin) & (freqs < fmax)
    bp = psd[..., band_mask].mean(axis=-1)  # mean over freqs
    return bp.mean(axis=1)                  # mean over channels -> (n_epochs,)


def _spectral_entropy(eeg: np.ndarray, sfreq: float, fmin=PSD_FMIN, fmax=PSD_FMAX):
    """
    Spectral entropy per epoch averaged over channels.
    """
    nperseg = int(sfreq * 1.0)
    freqs, psd = welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1, average="mean")
    band = (freqs >= fmin) & (freqs < fmax)
    psd_band = psd[..., band]
    psd_sum = psd_band.sum(axis=-1, keepdims=True) + 1e-12
    p = psd_band / psd_sum
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)  # (epochs, channels)
    return ent.mean(axis=1)  # average across channels


def _zscore_within_subject(x: np.ndarray):
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd


def compute_features_and_ei(epochs: mne.Epochs):
    """Return DataFrame with per-window features and EI."""
    X = epochs.get_data(picks="eeg")  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    # bandpowers on frontal only
    frontal_picks = _pick_frontal(epochs)
    Xf = X[:, frontal_picks, :]

    theta = _bandpower(Xf, sfreq, *BANDS["theta"])
    alpha = _bandpower(Xf, sfreq, *BANDS["alpha"])
    beta = _bandpower(Xf, sfreq, *BANDS["beta"])
    lgam = _bandpower(Xf, sfreq, *BANDS["lgam"])
    sent = _spectral_entropy(Xf, sfreq, fmin=PSD_FMIN, fmax=PSD_FMAX)

    # Build table
    df = pd.DataFrame({
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "lgam": lgam,
        "sent": sent,
    })

    # z-score within subject over all windows
    for col in df.columns:
        df[f"z_{col}"] = _zscore_within_subject(df[col])

    # EI per window: z(-alpha) + z(beta) + z(lgam) + z(sent)
    df["z_neg_alpha"] = _zscore_within_subject(-df["alpha"].values)
    df["EI"] = df["z_neg_alpha"] + df["z_beta"] + df["z_lgam"] + df["z_sent"]
    return df


def process_file(fif_path: str):
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    raw = _preprocess_raw(raw)
    segments = _extract_clip_segments(raw)

    rows = []
    all_windows = []

    for clip_id, spans in segments.items():
        for (t0, t1) in spans:
            # Crop to the clip span
            seg = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
            epochs = _fixed_length_epochs(seg, duration=WIN_SEC, step=STEP_SEC)
            if len(epochs) == 0:
                continue
            dfw = compute_features_and_ei(epochs)
            dfw["clip_id"] = clip_id
            dfw["t_start"] = np.arange(len(dfw)) * STEP_SEC + t0
            all_windows.append(dfw)

    if not all_windows:
        return pd.DataFrame(), pd.DataFrame()

    windows = pd.concat(all_windows, ignore_index=True)

    # Per-clip summary
    clip_summary = (
        windows.groupby("clip_id")
        .agg(EI_mean=("EI", "mean"),
             EI_sd=("EI", "std"),
             n_windows=("EI", "size"))
        .reset_index()
        .sort_values("EI_mean", ascending=False)
    )
    # Rank clips
    clip_summary["rank"] = clip_summary["EI_mean"].rank(ascending=False, method="dense").astype(int)
    clip_summary = clip_summary.sort_values(["rank", "clip_id"])

    return clip_summary, windows


def find_subject_files(data_dir: str, subject: str):
    """
    Find FIF for a subject. Adjust the glob as needed to match your layout.
    """
    pats = [
        os.path.join(data_dir, subject, "**", "*.fif"),
        os.path.join(data_dir, f"*{subject}*", "**", "*.fif"),
        os.path.join(data_dir, "*.fif"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # Prefer raw files over epochs/ICA files
    files = [f for f in files if "raw" in os.path.basename(f).lower() or f.endswith(".fif")]
    if not files:
        raise FileNotFoundError(f"No FIF found for subject pattern '{subject}' in {data_dir}")
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root directory containing OpenMIIR FIF files")
    ap.add_argument("--subject", required=True, help="Subject id (e.g., S01)")
    ap.add_argument("--out", default="ai_clips.csv", help="Output CSV for per-clip EI")
    ap.add_argument("--out_windows", default="ai_windows.csv", help="Output CSV for per-window EI")
    args = ap.parse_args()

    files = find_subject_files(args.data_dir, args.subject)

    all_clip_summaries = []
    all_windows = []
    for fif in files:
        try:
            clip_df, win_df = process_file(fif)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(fif)}: {e}")
            continue
        if not clip_df.empty:
            clip_df["subject"] = args.subject
            all_clip_summaries.append(clip_df)
        if not win_df.empty:
            win_df["subject"] = args.subject
            win_df["file"] = os.path.basename(fif)
            all_windows.append(win_df)

    if not all_clip_summaries:
        raise RuntimeError("No clips processed. Check annotations and patterns.")

    clips = pd.concat(all_clip_summaries, ignore_index=True)
    windows = pd.concat(all_windows, ignore_index=True)

    # Normalize EI_mean per subject (optional: easier cross-file comparison)
    clips["EI_mean_z"] = clips.groupby("subject")["EI_mean"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12)
    )

    clips.to_csv(args.out, index=False)
    windows.to_csv(args.out_windows, index=False)

    print(f"Wrote per-clip EI to: {args.out}")
    print(f"Wrote per-window EI to: {args.out_windows}")
    print("Top clips:")
    print(clips.sort_values("EI_mean", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
