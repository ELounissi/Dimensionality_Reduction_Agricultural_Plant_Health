# Dimensionality Reduction for Agricultural Plant Health Prediction

This repository contains the data, code, and complete results for the study *Impact of dimensionality reduction on machine learning and deep learning predictions of agricultural plant health*.

The benchmark compares six feature representations with six machine learning and deep learning models for fungal disease classification and severity regression in carrot, lettuce, and onion crops.

## Repository contents

```text
run_analysis.py                 Command-line entry point
reproduce_paper_results.ipynb  Step-by-step notebook
dr_agri/                       Data, preprocessing, DR, and model code
analysis/                      Internal workflow, statistics, and figure code
data/                          Released crop and weather inputs
results/                       Complete paper results, tables, and figures
requirements.txt               Pinned Python dependencies
```

The top level has only two analysis entry points. Use the notebook for an interactive walkthrough or the Python script for an automated run.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/ELounissi/Dimensionality_Reduction_Agricultural_Plant_Health.git
cd Dimensionality_Reduction_Agricultural_Plant_Health
python -m venv .venv
```

Activate the environment.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Jupyter only if the notebook interface is needed:

```bash
python -m pip install jupyter
```

## Quick verification using the included results

The repository ships with all 2,160 run-level results. The following command audits the protocol and regenerates the statistical tests, tables, and six figures from `results/runs.csv`:

```bash
python run_analysis.py --from-existing
```

This is the fastest way to verify the paper outputs without fitting the models again.

## Full reproduction from the data

Run the complete analysis with ten deterministic seeds:

```bash
python run_analysis.py --jobs 8
```

The default output directory is `results/reproduced`. Change the worker count and output directory as needed:

```bash
python run_analysis.py --jobs 16 --results-dir results/my_run
```

The full run performs the following stages:

1. Audit the split and preprocessing protocol.
2. Fit every crop, task, DR, model, and seed combination.
3. Save every held-out test score and selected model parameter.
4. Calculate means and 95 percent confidence intervals across seeds.
5. Calculate the Wilcoxon tests and pairing win-loss tables.
6. Generate Figures 1 to 5 and the runtime comparison figure.
7. Verify seed coverage, unique result keys, scores, and output files.

The analysis is CPU compatible. `--jobs` controls independent worker processes, so the same entry point can be used on a workstation or inside a scheduler selected by the user. Runtime measurements depend on the hardware and worker count. Predictive scores use deterministic seeds and the pinned environment.

## Notebook

Open `reproduce_paper_results.ipynb` with Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab reproduce_paper_results.ipynb
```

The notebook:

- loads the analysis modules directly
- prints the dataset summary
- runs the leakage and split audit
- regenerates outputs from the included result-level CSV
- prints the main tables and statistical results
- displays every generated figure
- contains an optional final cell for a complete model rerun

## Evaluation protocol

One plant-day record is treated as one sample. The released plant-level random split is retained. The analysis does not claim testing on unseen farms or future dates.

- Classical models use a deterministic 80/20 train-test split.
- Classical hyperparameters are selected by five-fold cross-validation using the training partition only.
- MLP, LSTM, and GRU use deterministic 70/10/20 train-validation-test splits.
- Neural validation controls early stopping. The test partition is evaluated once.
- Median imputation, standardization, and fitted DR transformations use training data only.
- The same partitions are reused across methods within each seed.
- Reported values are means and 95 percent confidence intervals over seeds.
- PCA retains 95 percent of training-set variance.
- KernelPCA, Isomap, and MDS use 22 components.
- BOTCAST uses its fixed expert-defined feature set.

The protocol audit checks partition disjointness, deterministic splits, target exclusion, training-only transformation fitting, class support, and the requested split proportions.

## Data used by the analysis

| Crop | Raw plant rows | Modeling rows | Predictors |
|---|---:|---:|---:|
| Onion | 922 | 922 | 135 |
| Lettuce | 589 | 540 | 132 |
| Carrot | 770 | 399 | 134 |
| **Total** | **2,281** | **1,861** | |

The released carrot analysis excludes `FarmID == 0`, leaving 499 records. Matching plant and weather records by farm and date leaves 399 carrot records and 540 lettuce records. Onion retains all 922 records.

## Generated outputs

| Path | Description |
|---|---|
| `results/runs.csv` | All 2,160 seed-level held-out test results |
| `results/aggregated.csv` | Means, confidence intervals, timings, and run counts |
| `results/tables.md` | Main classification and regression result tables |
| `results/statistical_tests.csv` | Raw and Holm-adjusted Wilcoxon results |
| `results/statistical_tests.md` | Readable statistical tables |
| `results/pairing_win_loss.csv` | DR and model pairing counts |
| `results/timing_comparison.csv` | Runtime and speedup comparison |
| `results/figures/Figure_1.png` | Training and validation curves |
| `results/figures/Figure_2.png` | Classification comparison by DR method |
| `results/figures/Figure_3.png` | Regression comparison by DR method |
| `results/figures/Figure_4.png` | Classification comparison by model |
| `results/figures/Figure_5.png` | Regression comparison by model |
| `results/figures/Timing_Comparison.png` | Runtime with and without DR |
| `results/manifests/protocol_audit.json` | Machine-readable protocol audit |

The included result set contains ten seeds, zero failed experimental units, zero duplicate result keys, and zero missing scores.
