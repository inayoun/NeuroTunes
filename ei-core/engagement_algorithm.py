"""EEG engagement processing primitives for NeuroTunes.

This module provides utilities to discover subject recordings, load raw FIF
files, apply the preprocessing reference pipeline, and derive clip annotations
that will be consumed by downstream work packages.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd
import numpy as np
from mne.time_frequency import psd_array_welch


logger = logging.getLogger(__name__)


def _normalise_subject_codes(subject: str) -> Iterable[str]:
    """Yield subject code variants (e.g. `S01` → [`S01`, `P01`])."""

    subj = subject.strip()
    if not subj:
        return []

    subj = subj.upper()
    digits = subj[1:] if len(subj) > 1 else ""

    variants = {subj}
    if subj.startswith("S") and digits.isdigit():
        variants.add(f"P{digits}")
        variants.add(digits)
    elif subj.startswith("P") and digits.isdigit():
        variants.add(f"S{digits}")
        variants.add(digits)
    elif subj.isdigit():
        variants.add(f"S{subj.zfill(2)}")
        variants.add(f"P{subj.zfill(2)}")

    return variants


def find_subject_files(data_root: str, subject: str) -> list[str]:
    """Locate raw FIF files for a subject within ``data_root``.

    The OpenMIIR distribution uses ``PXX-raw.fif`` naming. We accept either
    ``SXX`` or ``PXX`` identifiers and traverse subdirectories to find
    candidate files. Results are returned as absolute paths sorted
    lexicographically.
    """

    base = Path(data_root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {base}")

    candidates: set[Path] = set()
    for code in _normalise_subject_codes(subject):
        if not code:
            continue
        patterns = [
            f"**/{code}*-raw.fif",
            f"**/{code}*-raw_sss.fif",
            f"**/{code}-raw.fif",
            f"**/{code}.fif",
        ]
        for pattern in patterns:
            candidates.update(base.glob(pattern))

    files = sorted(str(path) for path in candidates if path.is_file())
    return files


def load_raw(path: str) -> mne.io.BaseRaw:
    """Load an MNE Raw object from ``path`` with ``preload=True``."""

    fif_path = Path(path).expanduser().resolve()
    if not fif_path.exists():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    *,
    notch: float = 50.0,
    bp: tuple[float, float] = (1.0, 45.0),
    resamp: float = 256.0,
) -> mne.io.BaseRaw:
    """Apply NeuroTunes preprocessing pipeline to a raw recording.

    Steps: (1) set BioSemi64 montage if available, (2) notch filter around the
    specified frequency, (3) band-pass filter, (4) common-average reference,
    and (5) resample to ``resamp`` Hz.
    """

    proc = raw.copy()

    try:
        proc.set_montage("biosemi64", match_case=False, on_missing="warn")
    except (ValueError, RuntimeError):
        # Warn silently; montage may be unavailable or channel labels differ.
        proc.info["description"] = proc.info.get("description")

    if notch:
        proc.notch_filter(freqs=notch)

    l_freq, h_freq = bp
    proc.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    proc.set_eeg_reference("average", projection=False)

    if resamp:
        proc.resample(resamp)

    return proc


_CLIP_PATTERN = re.compile(r"(?:stimulus/)?(clip_\d+)", re.IGNORECASE)


def _decode_event(event_id: int) -> tuple[int | None, str]:
    """Map OpenMIIR event IDs to clip index and condition string."""

    if event_id >= 2000:
        return None, "other"
    if event_id >= 1000:
        return None, {1000: "audio", 1111: "noise"}.get(event_id, "other")

    stimulus_id = event_id // 10
    condition_code = event_id % 10
    condition = {
        1: "listen",
        2: "cued",
        3: "fix",
        4: "imag",
    }.get(condition_code, "other")
    return stimulus_id, condition


_METADATA_CACHE: dict[Path, dict[int, float]] = {}


def _load_stimuli_metadata(subject_code: str, raw_path: Path) -> dict[int, float]:
    """Load stimulus durations for the given subject from metadata spreadsheets."""

    subject_code = subject_code.upper()
    digits = "".join(filter(str.isdigit, subject_code))
    version = "v1"
    if digits and int(digits) >= 9:
        version = "v2"

    dataset_root = raw_path.parent.parent.parent
    metadata_path = dataset_root / "meta" / f"Stimuli_Meta.{version}.xlsx"

    if metadata_path in _METADATA_CACHE:
        return _METADATA_CACHE[metadata_path]

    if not metadata_path.exists():
        return {}

    table = pd.read_excel(metadata_path, engine="openpyxl")
    if "id" not in table or "length of song (sec)" not in table:
        return {}

    mapping: dict[int, float] = {}
    for _, row in table.iterrows():
        stim_id = row.get("id")
        duration = row.get("length of song (sec)")
        if pd.isna(stim_id) or pd.isna(duration):
            continue
        stim_key = int(round(float(stim_id)))
        duration_val = float(duration)
        if duration_val > 0:
            mapping[stim_key] = duration_val

    _METADATA_CACHE[metadata_path] = mapping
    return mapping


def _annotations_from_events(raw: mne.io.BaseRaw) -> mne.Annotations | None:
    """Derive annotations from stimulus channel events if missing."""

    stim = None
    for name in raw.ch_names:
        if name.upper().startswith("STI"):
            stim = name
            break

    if stim is None:
        return None

    events = mne.find_events(raw, stim_channel=stim, verbose=False)
    if events.size == 0:
        return None

    sfreq = raw.info["sfreq"]

    raw_path = Path(raw.filenames[0]).resolve() if raw.filenames else None
    subject_code = raw_path.stem.split("-")[0] if raw_path else ""
    metadata = _load_stimuli_metadata(subject_code, raw_path) if raw_path else {}

    annotations = []
    for sample, _, event_id in events:
        clip_idx, condition = _decode_event(event_id)
        if clip_idx is None or condition != "listen":
            continue

        duration = metadata.get(clip_idx)
        if not duration or duration <= 0:
            continue

        onset = sample / sfreq
        annotations.append((onset, duration, f"clip_{clip_idx:02d}"))

    if not annotations:
        return None

    onsets, durations, descriptions = zip(*annotations)
    return mne.Annotations(onsets, durations, descriptions)


def extract_clip_segments(raw: mne.io.BaseRaw) -> dict[str, list[tuple[float, float]]]:
    """Collect annotation segments keyed by clip identifier.

    The annotations must contain ``clip_XX`` labels (optionally prefixed with
    ``Stimulus/``). Imagery trials (descriptions containing ``imag``) and
    non-positive durations are ignored.
    """

    segments: dict[str, list[tuple[float, float]]] = {}
    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        annotations = _annotations_from_events(raw)
    if annotations is None:
        return segments

    skipped_durations = 0

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if duration is None or not np.isfinite(duration) or duration <= 0:
            skipped_durations += 1
            continue

        desc_lower = description.lower()
        if "imag" in desc_lower:
            continue

        match = _CLIP_PATTERN.search(desc_lower)
        if not match:
            continue

        clip_id = match.group(1).lower()
        t0 = float(onset)
        t1 = t0 + float(duration)
        if t1 <= t0:
            continue

        segments.setdefault(clip_id, []).append((t0, t1))

    if skipped_durations:
        logger.warning(
            "Skipped %d annotations with missing duration for subject data.",
            skipped_durations,
        )

    return segments


def make_fixed_epochs(
    raw: mne.io.BaseRaw,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
) -> mne.Epochs:
    """Generate fixed-length epochs annotated with clip membership."""

    if win_sec <= 0:
        raise ValueError("win_sec must be positive")
    if step_sec <= 0:
        raise ValueError("step_sec must be positive")

    overlap = max(0.0, win_sec - step_sec)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=win_sec,
        overlap=overlap,
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    sfreq = raw.info["sfreq"]
    starts = epochs.events[:, 0] / sfreq + epochs.tmin
    stops = starts + win_sec

    segments = extract_clip_segments(raw)

    clip_ids: list[str | None] = []
    for mid in (starts + stops) / 2.0:
        clip = None
        for clip_id, intervals in segments.items():
            if any(t0 <= mid <= t1 for t0, t1 in intervals):
                clip = clip_id
                break
        clip_ids.append(clip)

    metadata = pd.DataFrame(
        {
            "window_index": np.arange(len(starts)),
            "t0": starts,
            "t1": stops,
            "clip_id": clip_ids,
        }
    )
    epochs.metadata = metadata
    return epochs


def _pick_frontal_channels(epochs: mne.Epochs, frontal: list[str]) -> list[int]:
    available = {ch.upper(): ch for ch in epochs.ch_names}
    missing = [ch for ch in frontal if ch.upper() not in available]
    if missing:
        raise ValueError(
            "Missing required frontal channels: " + ", ".join(sorted(missing))
        )
    picks = [epochs.ch_names.index(available[ch.upper()]) for ch in frontal]
    return picks


def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=float)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def _spectral_entropy(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=float)
    band = psd[:, mask]
    total = band.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    probs = band / total
    return -(probs * np.log(probs + np.finfo(float).eps)).sum(axis=1)


def compute_features(
    epochs: mne.Epochs,
    frontal: list[str] | None = None,
) -> pd.DataFrame:
    """Compute frontal band powers and spectral entropy for each epoch."""

    if frontal is None:
        frontal = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"]

    picks = _pick_frontal_channels(epochs, frontal)
    data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]

    n_fft = int(round(sfreq * 2))
    psd, freqs = psd_array_welch(
        data,
        sfreq,
        fmin=4.0,
        fmax=45.0,
        n_fft=n_fft,
        average="mean",
        verbose="ERROR",
    )
    psd_mean = psd.mean(axis=1)

    theta = _band_power(psd_mean, freqs, 4.0, 7.0)
    alpha = _band_power(psd_mean, freqs, 8.0, 13.0)
    beta = _band_power(psd_mean, freqs, 13.0, 30.0)
    lgam = _band_power(psd_mean, freqs, 30.0, 45.0)
    sent = _spectral_entropy(psd_mean, freqs, 4.0, 45.0)

    features = pd.DataFrame(
        {
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "lgam": lgam,
            "sent": sent,
        }
    )

    if epochs.metadata is not None:
        meta = epochs.metadata.reset_index(drop=True)
        features = pd.concat([meta, features], axis=1)

    features = features[features["clip_id"].notna()].reset_index(drop=True)
    return features


def _zscore(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros_like(values), index=series.index, dtype=float)
    return pd.Series((values - mean) / std, index=series.index, dtype=float)


def compute_ei(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored metrics and composite engagement index."""

    df = features_df.copy()
    df["z_alpha"] = _zscore(-df["alpha"])
    df["z_beta"] = _zscore(df["beta"])
    df["z_lgam"] = _zscore(df["lgam"])
    df["z_sent"] = _zscore(df["sent"])
    df["EI"] = df[["z_alpha", "z_beta", "z_lgam", "z_sent"]].sum(axis=1)
    return df


def aggregate_clips(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window-level EI into clip summaries."""

    if windows_df.empty:
        return pd.DataFrame(columns=["clip_id", "EI_mean", "EI_sd", "n_windows", "rank"])

    df = windows_df.dropna(subset=["clip_id"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["clip_id", "EI_mean", "EI_sd", "n_windows", "rank"])

    grouped = df.groupby("clip_id", as_index=False).agg(
        EI_mean=("EI", "mean"),
        EI_sd=("EI", "std"),
        n_windows=("EI", "count"),
    )
    grouped["EI_sd"] = grouped["EI_sd"].fillna(0.0)
    grouped["rank"] = grouped["EI_mean"].rank(
        ascending=False, method="dense"
    ).astype(int)
    grouped = grouped.sort_values("rank").reset_index(drop=True)
    return grouped


def score_subject(data_root: str, subject: str) -> dict | None:
    """Compute quality score for a subject per WP3 specification."""

    files = find_subject_files(data_root, subject)
    if not files:
        return None

    total_windows = 0
    bad_windows = 0
    clip_durations: dict[str, float] = {}

    for path in files:
        try:
            raw = load_raw(path)
            proc = preprocess_raw(raw)
        except Exception:
            continue

        segments = extract_clip_segments(proc)
        for clip_id, intervals in segments.items():
            duration = sum(max(0.0, t1 - t0) for t0, t1 in intervals)
            if duration > 0:
                clip_durations[clip_id] = clip_durations.get(clip_id, 0.0) + duration

        try:
            epochs = make_fixed_epochs(proc)
        except Exception:
            continue

        data = epochs.get_data()
        if data.size == 0:
            continue

        ptp = np.ptp(data, axis=2)
        max_ptp = ptp.max(axis=1)
        bad_windows += int(np.sum(max_ptp > 150e-6))
        total_windows += len(max_ptp)

    if total_windows == 0 or not clip_durations:
        return None

    bad_ratio = bad_windows / total_windows
    clips_with_durations = sum(1 for dur in clip_durations.values() if dur > 0)
    score = clips_with_durations - 5 * bad_ratio

    return {
        "subject": subject,
        "clips_with_durations": clips_with_durations,
        "bad_window_ratio": bad_ratio,
        "score": score,
        "total_windows": total_windows,
        "bad_windows": bad_windows,
    }


def pick_subjects(data_root: str, candidates: list[str], k: int = 3) -> pd.DataFrame:
    """Rank subjects according to `score_subject` and return top ``k``."""

    results = []
    for subj in candidates:
        info = score_subject(data_root, subj)
        if info is None:
            continue
        results.append(info)

    if not results:
        return pd.DataFrame(columns=[
            "subject",
            "clips_with_durations",
            "bad_window_ratio",
            "score",
            "total_windows",
            "bad_windows",
        ])

    df = pd.DataFrame(results)
    df = df.sort_values(
        by=["score", "bad_window_ratio", "clips_with_durations"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return df.head(k)
