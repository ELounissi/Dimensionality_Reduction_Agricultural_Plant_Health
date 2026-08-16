"""Data loading and sequence construction.

One row of the returned design matrix is one plant-day observation. Its
predictors are that plant's own measurements on the sampling day together with
the weather variables of the same day and their backward-looking rolling
aggregates. No value posterior to the sampling day ever enters the predictors.
The plot context columns (FarmID, Plant_ID, Date) are retained as features, as
in the released data description: the study predicts disease for known,
monitored plots and does not claim generalization to unseen farms or future
dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

#: per-crop configuration: target column, the other disease columns that must be
#: dropped so they cannot act as predictors, and the crop-specific plant traits.
CROPS = {
    "Onion": dict(
        target="cote_b_squamosa",
        other_targets=["cote_p_destructor", "cote_s_vesicarium"],
        plant_features=["LivingLeavesNum_onions", "DeadLeavesNum_onions", "FeuillageCouche_onions"],
        drop_extra=["Bulb_onions_date"],
        max_score=4.0,
    ),
    "Lettuce": dict(
        target="cote_b_lactucae",
        other_targets=["incidence_sclerotinia", "incidence_b_cinerea"],
        plant_features=[],
        drop_extra=["Pommaison_lettuce_date"],
        max_score=4.0,
    ),
    "Carrot": dict(
        target="cote_c_carotae",
        other_targets=["incidence_a_dauci", "incidence_s_sclerotiorum"],
        plant_features=["GreenLeavesNum_carrots", "carrot_stage"],
        drop_extra=[],
        max_score=4.0,
    ),
}

_INDEX_COLS = ("Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y")


@dataclass
class CropData:
    """Container for one crop, one task."""

    crop: str
    task: str
    X: np.ndarray            # (n_obs, n_features) day-of-sample design matrix
    y: np.ndarray            # (n_obs,) target
    groups: np.ndarray       # (n_obs,) FarmID, retained for provenance summaries
    feature_names: list[str]
    plant_idx: np.ndarray    # indices of plant-trait columns inside X
    meteo_idx: np.ndarray    # indices of weather columns inside X
    static_idx: np.ndarray   # FarmID, Plant_ID and Date context columns
    frame: pd.DataFrame      # the merged frame, kept for sequence construction

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def load_crop(crop: str, task: str = "classification", data_root: Path | None = None) -> CropData:
    """Load one crop and return its design matrix and target.

    Parameters
    ----------
    crop : one of ``CROPS``
    task : ``"classification"`` (disease occurrence, binary) or ``"regression"``
           (disease severity, the ordinal score rescaled to ``[0, 1]``)
    """
    if crop not in CROPS:
        raise KeyError(f"unknown crop {crop!r}, expected one of {sorted(CROPS)}")
    if task not in {"classification", "regression"}:
        raise ValueError(f"unknown task {task!r}")

    cfg = CROPS[crop]
    root = Path(data_root) if data_root is not None else DATA_ROOT
    obs = pd.read_csv(root / crop / "crop_no_sensitive_data.csv").rename(columns={"SampleDate": "Date"})
    meteo = pd.read_csv(root / crop / "combined_daily_meteo.csv")

    # The released carrot analysis excludes FarmID 0. Retain that protocol.
    if crop == "Carrot":
        obs = obs[obs["FarmID"] != 0].copy()

    merged = obs.merge(meteo, on=["FarmID", "Date"], how="inner", suffixes=("", "_meteo"))
    merged = merged.sort_values(["FarmID", "Plant_ID", "Date"]).reset_index(drop=True)

    target = merged[cfg["target"]].astype(float).to_numpy()
    if task == "classification":
        y = (target >= 1).astype(np.float32)
    else:
        y = (target / cfg["max_score"]).astype(np.float32)

    drop = [cfg["target"], *cfg["other_targets"], *cfg["drop_extra"]]
    design = merged.drop(columns=[c for c in drop if c in merged.columns])
    design = design.select_dtypes(include=[np.number])
    # Drop CSV index artefacts only. FarmID, Plant_ID and the date stamp are
    # retained as context features, exactly as in the released data
    # description: the study predicts disease for known, monitored plots, and
    # does not claim generalization to unseen farms or future dates.
    design = design.drop(columns=[c for c in design.columns
                                  if c in _INDEX_COLS or str(c).startswith("Unnamed")])
    design = design.loc[:, design.notna().any()]          # drop all-NaN columns
    # Keep missing values here.  Median imputation is a fitted preprocessing
    # operation and is therefore performed inside each training/CV split in
    # ``evaluate.py``.  Computing medians here would let held-out observations
    # influence the training representation.
    # constant columns are kept so that the feature count matches the published
    # description (129 weather variables plus the crop-specific plant traits);
    # StandardScaler leaves them at zero and they carry no information

    feature_names = list(design.columns)
    plant_set = set(cfg["plant_features"])
    plant_idx = np.array([i for i, c in enumerate(feature_names) if c in plant_set], dtype=int)
    static_set = {"FarmID", "Plant_ID", "Date"}
    static_idx = np.array([i for i, c in enumerate(feature_names) if c in static_set], dtype=int)
    meteo_columns = set(meteo.select_dtypes(include=[np.number]).columns)
    meteo_idx = np.array([i for i, c in enumerate(feature_names)
                          if c in meteo_columns and c not in static_set], dtype=int)

    return CropData(
        crop=crop,
        task=task,
        X=design.to_numpy(dtype=np.float64),
        y=y,
        groups=merged["FarmID"].to_numpy(),
        feature_names=feature_names,
        plant_idx=plant_idx,
        meteo_idx=meteo_idx,
        static_idx=static_idx,
        frame=merged,
    )


def build_sequences(data: CropData, seq_len: int, data_root: Path | None = None) -> np.ndarray:
    """Build ``(n_obs, seq_len, n_features)`` backward-looking sequences.

    For an observation sampled on day ``t`` in farm ``f``, time step ``k`` of the
    sequence carries the weather variables of farm ``f`` on day ``t - seq_len + 1 + k``.
    Plant traits are measured only on the sampling day and are therefore repeated
    along the window. Days before the first available weather record are padded
    with the earliest record of that farm, which is the standard causal
    (backward-fill) padding and never introduces future information.

    ``seq_len=1`` reproduces the single-step formulation.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if seq_len == 1:
        return data.X[:, None, :]

    root = Path(data_root) if data_root is not None else DATA_ROOT
    meteo = pd.read_csv(root / data.crop / "combined_daily_meteo.csv")
    names = data.feature_names
    meteo_names = [names[i] for i in data.meteo_idx]

    # lookup table: (FarmID, Date) -> weather feature vector
    meteo = meteo.set_index(["FarmID", "Date"]).sort_index()
    missing = [c for c in meteo_names if c not in meteo.columns]
    if missing:
        raise KeyError(f"weather columns absent from the meteo table: {missing[:5]}")
    table = meteo[meteo_names]
    lookup = {key: row for key, row in zip(table.index, table.to_numpy(dtype=np.float64))}
    earliest = {f: sub.index.get_level_values("Date").min() for f, sub in table.groupby(level=0)}

    n_obs, n_feat = data.X.shape
    seq = np.repeat(data.X[:, None, :], seq_len, axis=1)   # plant traits repeated
    farms = data.frame["FarmID"].to_numpy()
    dates = data.frame["Date"].to_numpy()

    for i in range(n_obs):
        farm, day = farms[i], dates[i]
        first = earliest.get(farm, day)
        for k in range(seq_len):
            d = day - (seq_len - 1 - k)
            vec = lookup.get((farm, max(d, first)))
            if vec is None:                                 # gap in the record
                vec = lookup.get((farm, day))
            if vec is not None:
                seq[i, k, data.meteo_idx] = vec
    return seq


def dataset_summary(data_root: Path | None = None) -> pd.DataFrame:
    """Row counts actually produced by the pipeline, for reporting in the paper."""
    rows = []
    for crop in CROPS:
        d = load_crop(crop, "classification", data_root)
        rows.append(dict(crop=crop, observations=len(d.y), features=d.n_features,
                         fields=int(pd.Series(d.groups).nunique()),
                         positives=int(d.y.sum()),
                         positive_rate=round(float(d.y.mean()), 4)))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = dict(crop="TOTAL", observations=out["observations"].sum(),
                             features="", fields=out["fields"].sum(),
                             positives=out["positives"].sum(), positive_rate="")
    return out
