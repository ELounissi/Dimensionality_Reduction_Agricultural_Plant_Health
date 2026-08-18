"""Deterministic data partitions shared by every model.

One row of the modeling table is one plant-day observation, and the outer
split is a deterministic, seed-reproducible split of whole plants: every
observation of a plant falls entirely on one side, 80% of the plants for
training and selection, 20% held out for the single final evaluation. Class
proportions are balanced across the split for classification.

Grouping by plant keeps every visit history intact on both sides. A
training row's sequence is built entirely from observations of the same
plant, which are all in the training partition, so no held-out observation
ever reaches a fitting input and the fitted weights depend on the training
partition alone. A scored row likewise keeps the full history a deployed
model would hold for that plant. Sequences are therefore never truncated:
isolation and full temporal context hold simultaneously, which is not
possible when a split cuts through a plant's timeline.

Only the cross-plant channel needs scoping: the inverse-distance neighbor
severity of a sweep is aggregated over the fitting partition's plants when
building fitting inputs (``allowed`` in pipeline.Cell.sequences).

Neural models carve a further validation share out of the training
partition, again by whole plants, to drive early stopping.

The reported quantity is generalization to plants never observed during
fitting, within the fields and the season of the study; no claim is made
about unseen farms or future seasons.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     StratifiedGroupKFold)

TEST_SIZE = 0.20
VAL_FRACTION_OF_TRAIN = 0.125      # 0.8 * 0.125 = 0.10 -> a 70/10/20 partition
N_FOLDS = 5


def _outer_split(y, task: str, seed: int, groups: np.ndarray):
    """Reproducible plant-level 80/20 split of the plant-day observations,
    class-balanced for classification; identical for every model at a given
    seed."""
    idx = np.arange(len(y))
    if task == "classification":
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)
    else:
        cv = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                               random_state=seed)
    tr, te = next(iter(cv.split(idx, y, groups)))
    return idx[tr], idx[te]


def _val_split(tr: np.ndarray, y, seed: int, groups: np.ndarray):
    """Early-stopping validation share, carved out by whole plants."""
    cv = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION_OF_TRAIN,
                           random_state=seed)
    a, b = next(iter(cv.split(tr, y[tr], groups[tr])))
    return tr[a], tr[b]


def _folds(tr: np.ndarray, y, task: str, seed: int, groups: np.ndarray):
    """Five grouped selection folds inside the training partition."""
    if task == "classification":
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                  random_state=seed)
    else:
        cv = GroupKFold(n_splits=N_FOLDS)
    return list(cv.split(tr, y[tr], groups[tr]))
