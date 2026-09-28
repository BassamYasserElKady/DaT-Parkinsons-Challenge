"""Parameterized multi-sample spatial-transcriptomics calibration skeleton."""

from .manifest import SampleRecord, load_manifest, validate_manifest
from .preprocess import ProcessedSample, process_sample

__all__ = [
    "SampleRecord",
    "ProcessedSample",
    "load_manifest",
    "validate_manifest",
    "process_sample",
]
