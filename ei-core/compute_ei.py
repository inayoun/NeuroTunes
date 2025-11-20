"""Command-line interface to compute NeuroTunes engagement artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _ensure_repo_path() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_path()

from engagement_algorithm import (  # noqa: E402  (import after path tweak)
    aggregate_clips,
    compute_ei,
    compute_features,
    extract_clip_segments,
    find_subject_files,
    load_raw,
    make_fixed_epochs,
    pick_subjects,
    preprocess_raw,
)


def _discover_subjects(data_root: Path) -> list[str]:
    subjects: set[str] = set()
    for fif in data_root.rglob("*-raw.fif"):
        subjects.add(fif.stem.split("-")[0])
    return sorted(subjects)


def _select_subject(data_root: Path, subject: str) -> tuple[str, pd.DataFrame | None]:
    if subject.lower() != "auto":
        return subject, None

    candidates = _discover_subjects(data_root)
    table = pick_subjects(str(data_root), candidates, k=3)
    if table.empty:
        raise RuntimeError("No valid subjects found for auto selection")
    chosen = table.iloc[0]["subject"]
    return str(chosen), table


def _process_subject(
    data_root: Path,
    subject: str,
    frontal: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = find_subject_files(str(data_root), subject)
    if not files:
        raise FileNotFoundError(f"No FIF files discovered for subject {subject}")

    windows_list: list[pd.DataFrame] = []

    for path in files:
        raw = load_raw(path)
        raw = preprocess_raw(raw)

        segments = extract_clip_segments(raw)
        if not segments:
            continue

        epochs = make_fixed_epochs(raw)
        features = compute_features(epochs, frontal=frontal)
        if features.empty:
            continue

        windows = compute_ei(features)
        windows.insert(0, "subject", subject)
        windows.insert(1, "source_file", Path(path).name)
        windows_list.append(windows)

    if not windows_list:
        raise RuntimeError(f"Subject {subject} produced no window data")

    windows_df = pd.concat(windows_list, ignore_index=True)
    clips_df = aggregate_clips(windows_df)
    clips_df.insert(0, "subject", subject)
    return windows_df, clips_df


def _write_outputs(out_dir: Path, windows: pd.DataFrame, clips: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_path = out_dir / "ai_windows.csv"
    clips_csv_path = out_dir / "ai_clips.csv"
    clips_json_path = out_dir / "ai_clips.json"

    windows.to_csv(windows_path, index=False)
    clips.to_csv(clips_csv_path, index=False)
    clips.to_json(clips_json_path, orient="records", indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute engagement artifacts")
    parser.add_argument(
        "--data_dir",
        default="openmiir/eeg/mne",
        help="Directory containing OpenMIIR raw FIF files",
    )
    parser.add_argument(
        "--subject",
        default="auto",
        help="Subject code (e.g., P01) or 'auto' to select best",
    )
    parser.add_argument(
        "--out_dir",
        default="ei-core/out",
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--frontal",
        default=None,
        help="Comma-separated list of frontal channel labels to override defaults",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.exists():
        parser.error(f"data_dir not found: {data_root}")

    subject, ranking_table = _select_subject(data_root, args.subject)
    print(f"Selected subject: {subject}")
    if ranking_table is not None:
        print("Top candidates:")
        print(ranking_table.to_string(index=False))

    frontal = (
        [label.strip() for label in args.frontal.split(",") if label.strip()]
        if args.frontal
        else None
    )

    windows_df, clips_df = _process_subject(data_root, subject, frontal=frontal)
    _write_outputs(Path(args.out_dir), windows_df, clips_df)

    print(f"Wrote {len(windows_df)} windows across {len(clips_df)} clips")
    print(f"Clip summary saved to {args.out_dir}/ai_clips.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
