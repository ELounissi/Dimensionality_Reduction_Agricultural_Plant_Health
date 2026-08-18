# Machine Learning and Dimensionality Reduction for Plant-Health Prediction

A complete, reproducible benchmark of **eight learning approaches** across
**six feature representations**, **three vegetable crops**, and **two
prediction tasks**, built on weekly in-field plant observations, daily
weather records, and plant-to-plant distances collected over a full growing
season.

The comparison spans four model families, each consuming the data
representation it is designed for:

| Family | Models | Input |
|---|---|---|
| Tree and neighbor methods | Decision tree, Random forest, k-NN | sampling-day feature vector |
| Feed-forward network | MLP | sampling-day feature vector |
| Pretrained tabular foundation model | TabPFN (v3) | sampling-day feature vector |
| Sequence models | LSTM, RNN GRU, Mamba SSM | the plant's recent visit history |

Representations: **No Reduction**, **PCA**, **KernelPCA**, **Isomap**,
**MDS**, and **BOTCAST** (an expert-defined agronomic feature set). Tasks:
disease-occurrence classification (F1) and disease-severity regression
(R2), each scored over ten deterministic seeds.

## Highlights

- **Rigorous protocol with a leakage-free fitting pipeline.** Shared
  deterministic 80/20 splits per seed, holding out whole plants so that
  no held-out observation reaches any fitting input while every visit
  history stays intact; imputation, standardization, every reduction,
  selection, and early stopping fitted on training-side data only; the
  test partition is scored exactly once per run; one model per run.
  Inputs follow the standard one-step-ahead convention: predictions at
  day *t* may use any observation recorded before *t*, and never a
  row's own target.
- **Fair treatment for every model.** Classical models receive a
  cross-validated hyperparameter search, neural models a cross-validated
  architecture search with early stopping, and TabPFN its pretrained
  prior. Sequence models additionally consume the temporal structure the
  dataset was designed to expose - strictly backward-looking.
- **Everything ships with the code.** The full run-level results
  (2,880 runs with scores, timings, and selected configurations) and
  every generated table and figure.

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
  dr_agri/                 data loading, reductions, classical model grids
data/<Crop>/             observations, daily weather, BOTCAST variables,
                         and plant-pair distances per crop
data/Data_description_Stephane.pdf  the dataset datasheet: collection
                         process, variables, and label definitions
results/                 the shipped benchmark run (runs.csv, win_loss.md)
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
python run_all.py          # the full benchmark, all defaults
python code/outputs.py     # all tables (CSV) and figures (PNG)
python code/report.py      # rankings + win/loss at a glance
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

`code/outputs.py` and `code/report.py` accept `--results` to point at
any output directory produced by `run_all.py`.

## Generated outputs

| File | Content |
|---|---|
| `outputs/tables/scores_<task>_<reduction>.csv` | mean, 95% CI, and best-in-crop marker per model and crop |
| `outputs/tables/win_loss_<task>.csv` | combined wins/losses per pairing with the best pairing flagged per column |
| `outputs/tables/rankings.csv` | overall model ranking per task |
| `outputs/tables/timing.csv` | mean experimental-unit time and speedup per representation |
| `outputs/figures/Figure_2..5.png` | boxplot comparisons by representation and by model |

## Experimental design notes

- **Partitions.** One row is one plant-day observation, and the outer
  80/20 split holds out whole plants: every observation of a plant
  falls on one side, class-balanced for classification and identical
  for every model at a given seed. The reported quantity is therefore
  generalization to plants never observed during fitting, within the
  fields and the season of the study; no claim is made about unseen
  farms or future seasons.
- **Training isolation with full history.** Because a plant's whole
  visit history sits on one side of the split, a training row's
  sequence is built entirely from training-side observations: no
  held-out observation reaches a fitting input, so the fitted weights
  depend on the training partition alone, and no sequence is truncated
  to achieve it. Scored rows keep the full history a deployed model
  would hold for that plant. The only cross-plant channel, the
  inverse-distance neighbor severity, is aggregated over the fitting
  partition when fitting inputs are built.
- **Selection.** Every model selects inside each seed's own training
  partition by five-fold cross-validation over grouped folds, so the
  plants of a selection fold are disjoint from those used to fit it:
  the classical models over their grids, and every neural model over
  its architecture, history length K where applicable, dropout rate,
  and learning rate. The early-stopping validation share is carved out
  by whole plants as well. No selection decision ever uses data outside
  the seed's own training partition.
- **Fixed training constants.** Weight decay, batch size, the
  projection-and-head layout, and the early-stopping policy are fixed
  globally. Dropout and learning rate are selected inside each seed's
  training partition as part of the neural-model search.

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
