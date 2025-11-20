import pathlib

import numpy as np
import pandas as pd
import pytest

# Ensure project modules are importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
EI_CORE = ROOT / "ei-core"
if str(EI_CORE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(EI_CORE))

from engagement_algorithm import (  # noqa: E402
    aggregate_clips,
    compute_ei,
    compute_features,
    extract_clip_segments,
    find_subject_files,
    load_raw,
    make_fixed_epochs,
    preprocess_raw,
    score_subject,
)

DATA_ROOT = ROOT / "openmiir" / "eeg" / "mne"
SUBJECT = "P01"


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="OpenMIIR data not available")
def test_feature_shapes_and_finiteness():
    files = find_subject_files(str(DATA_ROOT), SUBJECT)
    raw = load_raw(files[0])
    proc = preprocess_raw(raw)
    epochs = make_fixed_epochs(proc)
    features = compute_features(epochs)

    assert {"theta", "alpha", "beta", "lgam", "sent"}.issubset(features.columns)
    numeric = features[["theta", "alpha", "beta", "lgam", "sent"]].to_numpy()
    assert np.isfinite(numeric).all()

    windows = compute_ei(features)
    for col in ["z_alpha", "z_beta", "z_lgam", "z_sent", "EI"]:
        assert col in windows.columns
        assert np.isfinite(windows[col].to_numpy()).all()


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="OpenMIIR data not available")
def test_extract_clip_segments_positive_durations():
    files = find_subject_files(str(DATA_ROOT), SUBJECT)
    raw = load_raw(files[0])
    proc = preprocess_raw(raw)
    segments = extract_clip_segments(proc)

    assert segments, "expected segments for subject"
    assert all(
        all(t1 > t0 for t0, t1 in seg_list) for seg_list in segments.values()
    )


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="OpenMIIR data not available")
def test_aggregate_clips_consistency():
    files = find_subject_files(str(DATA_ROOT), SUBJECT)
    raw = load_raw(files[0])
    proc = preprocess_raw(raw)
    epochs = make_fixed_epochs(proc)
    features = compute_features(epochs)
    windows = compute_ei(features)
    clips = aggregate_clips(windows)

    assert clips.shape[0] > 0
    assert clips["n_windows"].sum() == len(windows)
    assert clips["EI_mean"].is_monotonic_decreasing or clips.shape[0] == 1


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="OpenMIIR data not available")
def test_score_subject_returns_metrics():
    info = score_subject(str(DATA_ROOT), SUBJECT)
    assert info is not None
    assert info["clips_with_durations"] > 0
    assert 0 <= info["bad_window_ratio"] < 1
