from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent / "dataset" / "My_fatigue" / "My_fatigue"


def _natural_key(path: Path) -> list[Any]:
    """Sort S1.npy, S2.npy, ..., S10.npy by embedded numbers."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _session_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse session id from file name: {path.name}")
    return int(match.group(1))


def resolve_root(root: str | Path = DEFAULT_ROOT) -> Path:
    """Return the inner My_fatigue folder that contains data_DE/ and label/."""
    root = Path(root)
    if (root / "data_DE").is_dir() and (root / "label").is_dir():
        return root

    inner_root = root / "My_fatigue"
    if (inner_root / "data_DE").is_dir() and (inner_root / "label").is_dir():
        return inner_root

    raise FileNotFoundError(
        f"Cannot find data_DE/ and label/ under {root} or {inner_root}"
    )


def find_session_pairs(root: str | Path = DEFAULT_ROOT) -> list[tuple[Path, Path]]:
    """Return ordered (data_file, label_file) pairs."""
    root = resolve_root(root)
    data_dir = root / "data_DE"
    label_dir = root / "label"

    data_files = sorted(data_dir.glob("*.npy"), key=_natural_key)
    label_files = sorted(label_dir.glob("*.npy"), key=_natural_key)
    if not data_files:
        raise FileNotFoundError(f"No .npy files found under: {data_dir}")
    if not label_files:
        raise FileNotFoundError(f"No .npy files found under: {label_dir}")

    data_by_id = {_session_id(path): path for path in data_files}
    label_by_id = {_session_id(path): path for path in label_files}
    if set(data_by_id) != set(label_by_id):
        missing_labels = sorted(set(data_by_id) - set(label_by_id))
        missing_data = sorted(set(label_by_id) - set(data_by_id))
        raise FileNotFoundError(
            f"Data/label session ids do not match. "
            f"Missing labels: {missing_labels}; missing data: {missing_data}"
        )

    return [(data_by_id[index], label_by_id[index]) for index in sorted(data_by_id)]


def load_session(
    data_path: str | Path,
    label_path: str | Path,
    dtype: np.dtype | type | None = np.float32,
    label_dtype: np.dtype | type | None = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one session as (data, label)."""
    data = np.load(data_path, allow_pickle=True)
    label = np.load(label_path, allow_pickle=True)

    data = np.asarray(data, dtype=dtype)
    label = np.asarray(label, dtype=label_dtype).squeeze()

    if data.shape[0] != label.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {data_path} has {data.shape[0]} samples, "
            f"but {label_path} has {label.shape[0]} labels."
        )

    return data, label


def load_subject(
    session_pairs: list[tuple[Path, Path]],
    subject_name: str,
    dtype: np.dtype | type | None = np.float32,
    label_dtype: np.dtype | type | None = np.float32,
    stack: bool = True,
) -> dict[str, Any]:
    """Load one subject from its three session files.

    The readme states that S1-S3 are subject 1, S4-S6 are subject 2, and so on.
    The loading process intentionally uses append for each session.
    """
    data_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    data_files: list[str] = []
    label_files: list[str] = []

    for data_path, label_path in session_pairs:
        data, label = load_session(
            data_path,
            label_path,
            dtype=dtype,
            label_dtype=label_dtype,
        )
        data_list.append(data)
        label_list.append(label)
        data_files.append(str(data_path))
        label_files.append(str(label_path))

    data_out: np.ndarray | list[np.ndarray]
    label_out: np.ndarray | list[np.ndarray]
    if stack:
        data_out = np.concatenate(data_list, axis=0)
        label_out = np.concatenate(label_list, axis=0)
    else:
        data_out = data_list
        label_out = label_list

    return {
        "data": data_out,
        "label": label_out,
        "data_files": data_files,
        "label_files": label_files,
        "sessions": [_session_id(Path(path)) for path in data_files],
        "subject_name": subject_name,
    }


def load_my_fatigue(
    root: str | Path = DEFAULT_ROOT,
    sessions_per_subject: int = 3,
    dtype: np.dtype | type | None = np.float32,
    label_dtype: np.dtype | type | None = np.float32,
    stack: bool = True,
    expected_subjects: int | None = 10,
    expected_samples_per_session: int | None = 200,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load My_fatigue as sub1, sub2, ...

    Example:
        subjects = load_my_fatigue()
        sub1 = subjects["sub1"]
        x = sub1["data"]      # (600, 31, 5)
        y = sub1["label"]     # (600,)
    """
    session_pairs = find_session_pairs(root)
    if len(session_pairs) % sessions_per_subject != 0:
        raise ValueError(
            f"{len(session_pairs)} sessions cannot be evenly split by "
            f"sessions_per_subject={sessions_per_subject}."
        )

    subject_count = len(session_pairs) // sessions_per_subject
    if expected_subjects is not None and subject_count != expected_subjects:
        message = f"Found {subject_count} subjects, expected {expected_subjects}."
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)

    subjects: dict[str, dict[str, Any]] = {}
    for subject_index in range(subject_count):
        start = subject_index * sessions_per_subject
        end = start + sessions_per_subject
        subject_key = f"sub{subject_index + 1}"
        subject = load_subject(
            session_pairs[start:end],
            subject_name=subject_key,
            dtype=dtype,
            label_dtype=label_dtype,
            stack=stack,
        )

        if expected_samples_per_session is not None:
            for data_path, _ in session_pairs[start:end]:
                data = np.load(data_path, allow_pickle=True)
                sample_count = data.shape[0]
                if sample_count != expected_samples_per_session:
                    message = (
                        f"{data_path.name} has {sample_count} samples, "
                        f"expected {expected_samples_per_session}."
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
        x: all samples
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
        label = subject["label"]
        if isinstance(data, list):
            data = np.concatenate(data, axis=0)
        if isinstance(label, list):
            label = np.concatenate(label, axis=0)

        labels = np.asarray(label, dtype=np.float32).squeeze()
        x_list.append(np.asarray(data))
        y_list.append(labels)
        group_list.append(np.full(labels.shape[0], group_id, dtype=np.int64))

    return (
        np.concatenate(x_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(group_list, axis=0),
    )


def _shape_of(value: np.ndarray | list[np.ndarray]) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        return value.shape
    if not value:
        return (0,)
    return (sum(item.shape[0] for item in value),) + value[0].shape[1:]


def _print_summary(subjects: dict[str, dict[str, Any]]) -> None:
    print(f"Loaded {len(subjects)} subjects.")
    for subject_key, subject in subjects.items():
        print(
            f"{subject_key}: sessions={subject['sessions']} | "
            f"data={_shape_of(subject['data'])} | "
            f"label={_shape_of(subject['label'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load My_fatigue .npy samples.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-stack", action="store_true", help="Keep each subject as session lists.")
    parser.add_argument("--strict", action="store_true", help="Raise if expected counts do not match.")
    args = parser.parse_args()

    subjects = load_my_fatigue(
        root=args.root,
        stack=not args.no_stack,
        strict=args.strict,
    )

    # Optional variable-style access in an interactive session:
    # globals().update(subjects)  # creates sub1, sub2, ...
    _print_summary(subjects)


if __name__ == "__main__":
    main()
