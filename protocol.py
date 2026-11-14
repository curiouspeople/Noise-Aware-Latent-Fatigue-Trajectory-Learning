"""Boundary-preserving primitives for future BSPC experiment runners.

Not an experiment entry point. Index intervals are half-open in sample units.
Raw EEG overlap/timestamp validity must be checked independently.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np

@dataclass(frozen=True)
class Span:
    subject: str
    session: str
    partition: str
    start: int
    stop: int


def temporal_spans(subject: str, session: str, n: int, train_ratio: float = .7,
                   validation_ratio: float = .1, gap: int = 0) -> tuple[Span, Span, Span]:
    """Split one session before windowing; discard gap samples at boundaries."""
    if n <= 0 or gap < 0 or not (0 < train_ratio < 1 and 0 < validation_ratio < 1):
        raise ValueError('invalid sample count, ratios or gap')
    if train_ratio + validation_ratio >= 1:
        raise ValueError('training and validation ratios must leave test samples')
    a, b = int(n * train_ratio), int(n * (train_ratio + validation_ratio))
    spans = (Span(subject, session, 'train', 0, a),
             Span(subject, session, 'validation', a + gap, b),
             Span(subject, session, 'test', b + gap, n))
    if any(x.start >= x.stop for x in spans):
        raise ValueError('empty partition; reduce gap or increase data')
    return spans


def windows(span: Span, length: int, stride: int) -> Iterator[Span]:
    """Generate windows only within one subject/session/partition span."""
    if length <= 0 or stride <= 0 or span.start < 0 or span.stop <= span.start:
        raise ValueError('invalid window or span')
    for start in range(span.start, span.stop - length + 1, stride):
        yield Span(span.subject, span.session, span.partition, start, start + length)


def training_targets(labels: np.ndarray, span: Span, smoothing_window: int = 1) -> np.ndarray:
    """Return isolated training targets; validation/test targets cannot be smoothed."""
    if span.partition != 'train':
        raise ValueError('training_targets only accepts a training span')
    x = np.asarray(labels)
    if x.ndim != 1 or span.start < 0 or span.stop > len(x) or span.start >= span.stop:
        raise ValueError('invalid label array or bounds')
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError('smoothing_window must be a positive odd integer')
    y = x[span.start:span.stop].astype(np.float64, copy=True)
    if smoothing_window > len(y):
        raise ValueError('smoothing window exceeds partition')
    if smoothing_window > 1:
        from scipy.ndimage import uniform_filter1d
        y = uniform_filter1d(y, size=smoothing_window, mode='reflect')
    return y


def check_disjoint_subjects(train, validation, test) -> None:
    """Fail if any participant occurs in more than one data partition."""
    groups = [set(train), set(validation), set(test)]
    if any(not group for group in groups):
        raise ValueError('every partition must contain subjects')
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('subject leakage across partitions')
