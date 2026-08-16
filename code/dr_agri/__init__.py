"""Leakage-free benchmarking of dimensionality reduction for plant disease forecasting."""
from .data import CROPS, CropData, build_sequences, dataset_summary, load_crop
from .dr import DR_METHODS, make_reducer
from .evaluate import (Result, confidence_interval, prepare_neural_view, run_classical,
                       run_classical_all, run_neural)
from .models import MODEL_ORDER

__all__ = ["CROPS", "CropData", "load_crop", "build_sequences", "dataset_summary",
           "DR_METHODS", "make_reducer", "MODEL_ORDER", "Result", "run_classical",
           "run_neural", "run_classical_all", "prepare_neural_view", "confidence_interval"]
__version__ = "2.0.0"
