from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


DEFAULT_ROOT = Path(__file__).resolve().parent / "dataset" / "seedvig"


def _natural_key(path: Path) -> list[Any]:
    """Sort file/folder names by embedded numbers instead of pure text order."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_subject_dirs(root: str | Path = DEFAULT_ROOT) -> list[Path]:
    """Return all subject folders under the SEED-VIG root directory."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    subject_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=_natural_key,
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found under: {root}")
    return subject_dirs


def load_mat_sample(
    mat_path: str | Path,
    data_key: str = "eeg_data",
    label_key: str = "label",
    dtype: np.dtype | type | None = np.float32,
) -> tuple[np.ndarray, float]:
    """Load one .mat sample as (eeg_data, label)."""
    mat_path = Path(mat_path)
    mat = loadmat(mat_path)

    missing_keys = [key for key in (data_key, label_key) if key not in mat]
    if missing_keys:
        raise KeyError(f"{mat_path} is missing key(s): {missing_keys}")

    data = np.asarray(mat[data_key], dtype=dtype)
    label_array = np.asarray(mat[label_key]).squeeze()
    if label_array.size != 1:
        raise ValueError(f"{mat_path} label should contain one value, got {label_array.shape}")

    return data, float(label_array)


def load_subject(
    subject_dir: str | Path,
    data_key: str = "eeg_data",
    label_key: str = "label",
    dtype: np.dtype | type | None = np.float32,
    stack: bool = True,
) -> dict[str, Any]:
    """Load one subject folder.

    The loading process intentionally uses append:
        data_list.append(data)
        label_list.append(label)

    Returns a dict with:
        data: np.ndarray with shape (samples, time_points, channels) if stack=True,
              otherwise a list of per-sample arrays.
        label: np.ndarray with shape (samples,)
        files: ordered .mat file paths
        subject_name: source folder name
    """
    subject_dir = Path(subject_dir)
    mat_files = sorted(subject_dir.glob("*.mat"), key=_natural_key)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {subject_dir}")

    data_list: list[np.ndarray] = []
    label_list: list[float] = []

    for mat_file in mat_files:
        data, label = load_mat_sample(
            mat_file,
            data_key=data_key,
            label_key=label_key,
            dtype=dtype,
        )
        data_list.append(data)
        label_list.append(label)

    data_out: np.ndarray | list[np.ndarray]
    data_out = np.stack(data_list, axis=0) if stack else data_list
    label_out = np.asarray(label_list, dtype=np.float32)

    return {
        "data": data_out,
        "label": label_out,
        "files": [str(path) for path in mat_files],
        "subject_name": subject_dir.name,
    }


def load_seedvig(
    root: str | Path = DEFAULT_ROOT,
    data_key: str = "eeg_data",
    label_key: str = "label",
    dtype: np.dtype | type | None = np.float32,
    stack: bool = True,
    expected_samples: int | None = 885,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load all SEED-VIG subject folders as sub1, sub2, ...

    Example:
        subjects = load_seedvig()
        sub1 = subjects["sub1"]
        x = sub1["data"]
        y = sub1["label"]
    """
    subjects: dict[str, dict[str, Any]] = {}

    for subject_index, subject_dir in enumerate(find_subject_dirs(root), start=1):
        subject_key = f"sub{subject_index}"
        subject = load_subject(
            subject_dir,
            data_key=data_key,
            label_key=label_key,
            dtype=dtype,
            stack=stack,
        )

        sample_count = len(subject["label"])
        if expected_samples is not None and sample_count != expected_samples:
            message = (
                f"{subject_key} ({subject_dir.name}) has {sample_count} samples, "
                f"expected {expected_samples}."
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)

        subjects[subject_key] = subject

    return subjects


def merge_subjects(
    subjects: dict[str, dict[str, Any]],
    subject_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge selected subjects for model input.

    Returns:
        x: all EEG samples
        y: all labels
        groups: integer subject id for each sample
    """
    if subject_keys is None:
        subject_keys = list(subjects.keys())

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []

    for group_id, subject_key in enumerate(subject_keys):
        subject = subjects[subject_key]
        data = subject["data"]
        if isinstance(data, list):
            data = np.stack(data, axis=0)

        labels = np.asarray(subject["label"], dtype=np.float32)
        x_list.append(data)
        y_list.append(labels)
        group_list.append(np.full(labels.shape[0], group_id, dtype=np.int64))

    return (
        np.concatenate(x_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(group_list, axis=0),
    )


def _print_summary(subjects: dict[str, dict[str, Any]]) -> None:
    print(f"Loaded {len(subjects)} subjects.")
    for subject_key, subject in subjects.items():
        data = subject["data"]
        data_shape = data.shape if isinstance(data, np.ndarray) else (len(data),) + data[0].shape
        label_shape = subject["label"].shape
        print(
            f"{subject_key}: {subject['subject_name']} | "
            f"data={data_shape} | label={label_shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load SEED-VIG .mat samples.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-stack", action="store_true", help="Keep each subject as sample lists.")
    parser.add_argument("--strict", action="store_true", help="Raise if a subject does not have 885 samples.")
    args = parser.parse_args()

    subjects = load_seedvig(
        root=args.root,
        stack=not args.no_stack,
        strict=args.strict,
    )

    # Optional variable-style access in an interactive session:
    # globals().update(subjects)  # creates sub1, sub2, ...
    _print_summary(subjects)


if __name__ == "__main__":
    main()
