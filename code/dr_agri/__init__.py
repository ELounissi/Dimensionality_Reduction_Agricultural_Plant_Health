"""Data loading, feature representations, partitions, and classical
model grids of the plant-health benchmark."""
from .data import CROPS, CropData, build_sequences, dataset_summary, load_crop
from .dr import DR_METHODS, make_reducer
from .models import CLASSICAL_GRIDS, MODEL_ORDER, make_classical
from .splits import TEST_SIZE, VAL_FRACTION_OF_TRAIN, _outer_split

__all__ = ["CROPS", "CropData", "load_crop", "build_sequences",
           "dataset_summary", "DR_METHODS", "make_reducer",
           "CLASSICAL_GRIDS", "MODEL_ORDER", "make_classical",
           "TEST_SIZE", "VAL_FRACTION_OF_TRAIN", "_outer_split"]
__version__ = "3.0.0"
