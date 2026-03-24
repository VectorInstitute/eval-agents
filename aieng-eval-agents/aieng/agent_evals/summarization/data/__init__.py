"""Data loading and management for knowledge QA evaluation.

This module provides tools for loading and managing benchmark datasets
like DeepSearchQA.
"""

from .deepsearchqa import DeepSearchQADataset, DSQAExample


__all__ = [
    "DSQAExample",
    "DeepSearchQADataset",
]
