# Machine Learning and Dimensionality Reduction for Plant-Health Prediction

A complete, reproducible benchmark of **eight learning approaches** across
**six feature representations (No Reduction and 5 Dimensionality Reductions)**, **three vegetable crops**, and **two
prediction tasks**, built on weekly in-field plant observations, daily
weather records, and plant-to-plant distances collected over a full growing
season.

The comparison spans four model families, each consuming the data
representation it is designed for:

| Family | Models |
|---|---|
| Tree and neighbor methods | Decision tree, Random forest, k-NN |
| Feed-forward network | MLP |
| Pretrained tabular foundation model | TabPFN (v3) |
| Sequence models | LSTM, RNN GRU, Mamba SSM |

Representations: **No Reduction**, **PCA**, **KernelPCA**, **Isomap**,
**MDS**, and **BOTCAST** (an expert-defined agronomic feature set). Tasks:
disease-occurrence classification (F1) and disease-severity regression
(R2), each scored over ten deterministic seeds.

## Highlights

- **Rigorous, leakage-free protocol.** Shared deterministic 80/20 splits
  per seed; imputation, standardization, and every reduction fitted on
  training rows only; the test partition is scored exactly once per run;
  one model per run.
- **Fair treatment for every model.** Classical models receive a
  cross-validated hyperparameter search, neural models a cross-validated
  architecture search with early stopping, and TabPFN its pretrained
  prior. Sequence models additionally consume the temporal structure the
  dataset was designed to expose - strictly backward-looking.
- **Everything ships with the code.** The full run-level results
  (2,880 runs with scores, timings, and selected configurations), the
  generated tables and figures, and the originally published baseline for
  direct comparison.

## Repository layout

```text
run_all.py               benchmark driver (fully parameterized, see below)
explore_benchmark.ipynb  guided, step-by-step notebook tour
requirements.txt         pinned dependencies
code/                    the benchmark implementation
  pipeline.py              shared feature and sequence construction
  nets.py                  MLP, LSTM, GRU, and Mamba architectures + training
  outputs.py               tables (CSV) and figures (PNG) from results
  report.py                quick rankings and win/loss summary (markdown)
  compare_published.py     comparison against the published baseline
  dr_agri/                 data loading, reductions, classical model grids
data/<Crop>/             observations, daily weather, BOTCAST variables,
                         and plant-pair distances per crop
data/Data_description_Stephane.pdf  the dataset datasheet: collection
                         process, variables, and label definitions
results/                 the shipped benchmark run (runs.csv)
outputs/                 generated tables (CSV) and figures (PNG)
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Python 3.11 is recommended. TabPFN downloads its checkpoints on first use
and asks for a one-time license acceptance (https://ux.priorlabs.ai);
cached checkpoints then run fully offline. Every other model runs out of
the box.

## Quick start

```bash
python run_all.py                      # the full benchmark, all defaults
python code/outputs.py                      # all tables (CSV) and figures (PNG)
python code/report.py                       # rankings + win/loss at a glance
python code/compare_published.py            # deltas against the published scores
```

## run_all.py parameters

Every parameter is optional; with no parameters the complete benchmark
runs. Any combination can be included or excluded.

| Parameter | Values | Default | Description |
|---|---|---|---|
| `--tasks` | `classification`, `regression` | both | prediction tasks to run |
| `--crops` | `Carrot`, `Lettuce`, `Onion` | all three | crops to include |
| `--dr` | `"No Reduction"`, `PCA`, `KernelPCA`, `Isomap`, `MDS`, `BOTCAST` | all six | representations to include |
| `--models` | `"Decision tree"`, `"Random forest"`, `k-NN`, `MLP`, `LSTM`, `"RNN GRU"`, `"Mamba SSM"`, `TabPFN` | all eight | models to include |
| `--exclude-models` | same values | none | remove models from the selected set |
| `--exclude-dr` | same values | none | remove representations from the selected set |
| `--seeds` | any integers | `0 1 2 3 4 5 6 7 8 9` | explicit seed values |
| `--n-jobs` | integer | CPU count | parallel worker processes |
| `--out` | path | `results` | output directory for `runs.csv` |

Multi-word values are quoted. Examples:

```bash
# Sequence models only, on onion, with PCA and BOTCAST, three seeds
python run_all.py --models LSTM "RNN GRU" "Mamba SSM" \
    --crops Onion --dr PCA BOTCAST --seeds 0 1 2

# Everything except TabPFN and MDS
python run_all.py --exclude-models TabPFN --exclude-dr MDS

# Regression only, single seed, custom output directory
python run_all.py --tasks regression --seeds 7 --out results_seed7

# Classical models on the raw representation, sixteen workers
python run_all.py --models "Decision tree" "Random forest" k-NN \
    --dr "No Reduction" --n-jobs 16
```

`code/outputs.py`, `code/report.py`, and `code/compare_published.py` accept `--results`
to point at any output directory produced by `run_all.py`.

## Generated outputs

| File | Content |
|---|---|
| `outputs/tables/scores_<task>_<reduction>.csv` | mean, 95% CI, and best-in-crop marker per model and crop |
| `outputs/tables/win_loss_<task>.csv` | combined wins/losses per pairing with the best pairing flagged per column |
| `outputs/tables/rankings.csv` | overall model ranking per task |
| `outputs/tables/timing.csv` | mean experimental-unit time and speedup per representation |
| `outputs/figures/Figure_2..5.png` | boxplot comparisons by representation and by model |
| `results/vs_published.csv` | cell-level comparison against the published baseline |

## Guided notebook

`explore_benchmark.ipynb` walks through the dataset and the benchmark step
by step: inspecting the three data files of each crop, visualizing a
reduction, assembling a plant's visit-history sequence, training a model
on a small slice, and regenerating the headline tables and figures from
the shipped results. It is designed to be read top to bottom and runs in a
few minutes.

## Data

Each crop ships with its three released files: per-plant field
observations (weekly visits with disease scores), daily weather with
derived agronomic variables, and plant-pair distances measured in the
field. The BOTCAST variable list defines the expert representation. One
row of the modeling table is one plant on one sampling day. The full
datasheet - collection process, weather variable catalogue, and the
per-crop label definitions - is included as
`data/Data_description_Stephane.pdf`.
