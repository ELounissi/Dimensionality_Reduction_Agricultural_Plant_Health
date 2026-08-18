# v5 win/loss reported per crop (instead of pooled over the three crops)

Same two-axis rule as the pooled table, applied within one crop: a cell can win
on the row axis (best reduction for that model) and on the column axis (best model
for that reduction), so records run from 2/0 to 0/-2.

## classification

### Carrot — best pairing: **BOTCAST + Random forest** (F1 = 0.7350)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 0/0 | 0/-1 | 0/-1 | 0/-2 | 0/-1 | 1/0 |
| Random forest | 0/0 | 0/-1 | 0/-1 | 0/0 | 0/0 | **2/0** |
| k-NN | 1/-2 | 1/-2 | 1/-1 | 1/-1 | 1/-2 | 1/-1 |
| MLP | 1/0 | 1/0 | 0/0 | 0/-2 | 1/0 | 0/0 |
| LSTM | 0/-1 | 1/0 | 0/0 | 0/-2 | 0/0 | 0/0 |
| RNN GRU | 0/-1 | 1/0 | 0/0 | 0/-1 | 1/0 | 0/-2 |
| Mamba SSM | 0/-1 | 1/0 | 0/-1 | 0/0 | 0/0 | 0/-2 |
| TabPFN | **2/0** | **2/0** | **1/0** | **1/-1** | **2/0** | 2/0 |

Column bests: No Reduction: TabPFN, PCA: TabPFN, KernelPCA: TabPFN, Isomap: TabPFN, MDS: TabPFN, BOTCAST: Random forest

### Lettuce — best pairing: **PCA + TabPFN** (F1 = 0.9001)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 0/-1 | 0/-1 | 0/-2 | 0/-2 | 0/-1 | **2/0** |
| Random forest | 2/-1 | 0/-1 | 0/-1 | 0/-1 | 2/0 | 2/0 |
| k-NN | 1/-2 | 1/-1 | 1/-1 | 1/-1 | 1/-2 | 1/-2 |
| MLP | 2/-1 | **2/0** | 2/-1 | 1/-1 | **2/0** | 2/0 |
| LSTM | 1/-2 | 2/-1 | 2/-1 | 1/-1 | 2/-1 | 2/-1 |
| RNN GRU | **2/0** | 2/0 | 1/0 | 0/-2 | 2/0 | 2/0 |
| Mamba SSM | 2/0 | 2/0 | 2/-1 | 0/-1 | 2/-2 | 2/0 |
| TabPFN | 2/-1 | 2/-1 | **2/-1** | **2/-1** | 2/-1 | 2/-1 |

Column bests: No Reduction: RNN GRU, PCA: MLP, KernelPCA: TabPFN, Isomap: TabPFN, MDS: MLP, BOTCAST: Decision tree

### Onion — best pairing: **Isomap + RNN GRU** (F1 = 0.8061)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 1/-1 | 1/-1 | 1/-1 | 0/-1 | 1/-1 | 0/-2 |
| Random forest | 0/-1 | 1/-1 | 1/-1 | 2/0 | 2/0 | 1/-1 |
| k-NN | **2/0** | 2/0 | 0/-2 | 2/0 | **2/0** | 0/-1 |
| MLP | 2/0 | 2/0 | **2/0** | 2/0 | 2/0 | 0/-1 |
| LSTM | 1/-1 | **2/0** | 2/-1 | 2/0 | 2/-1 | 1/-1 |
| RNN GRU | 1/-1 | 2/-1 | 2/-1 | **2/0** | 2/0 | 1/-1 |
| Mamba SSM | 0/-1 | 2/-1 | 2/-1 | 2/0 | 0/-2 | **2/-1** |
| TabPFN | 1/-1 | 1/-1 | 2/-1 | 2/0 | 0/-1 | 1/-1 |

Column bests: No Reduction: k-NN, PCA: LSTM, KernelPCA: MLP, Isomap: RNN GRU, MDS: k-NN, BOTCAST: Mamba SSM

**classification: 8 distinct winners across the 18 columns** — TabPFN ×7, MLP ×3, RNN GRU ×2, k-NN ×2, Random forest ×1, Decision tree ×1, LSTM ×1, Mamba SSM ×1

## regression

### Carrot — best pairing: **BOTCAST + TabPFN** (R2 = 0.2823)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 0/0 | 0/0 | 0/-1 | 0/-1 | 0/-1 | 1/0 |
| Random forest | **2/0** | 1/0 | 0/-1 | 0/-1 | 1/0 | 0/0 |
| k-NN | 0/0 | 0/0 | 0/0 | **1/0** | 1/0 | 0/-1 |
| MLP | 0/0 | 1/0 | 0/0 | 0/0 | 0/-1 | 1/0 |
| LSTM | 1/0 | 0/0 | 0/0 | 0/-2 | 1/0 | 0/-1 |
| RNN GRU | 0/-2 | 0/-1 | **1/0** | 0/-1 | 0/0 | 0/0 |
| Mamba SSM | 1/0 | 0/0 | 0/-1 | 0/-1 | 0/0 | 0/0 |
| TabPFN | 2/-1 | **2/-1** | 1/-1 | 1/-1 | **2/-1** | **2/0** |

Column bests: No Reduction: Random forest, PCA: TabPFN, KernelPCA: RNN GRU, Isomap: k-NN, MDS: TabPFN, BOTCAST: TabPFN

### Lettuce — best pairing: **KernelPCA + Random forest** (R2 = 0.7322)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 1/-1 | 1/-1 | 0/-1 | 0/-1 | 0/-1 | 0/-1 |
| Random forest | **2/0** | **2/0** | **2/0** | 1/0 | 1/0 | 0/-1 |
| k-NN | 2/0 | 1/0 | 1/0 | 2/0 | 2/0 | 0/-2 |
| MLP | 1/0 | 1/0 | 1/0 | 0/0 | **2/0** | 0/-1 |
| LSTM | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/-1 |
| RNN GRU | 1/0 | 1/0 | 1/0 | 0/-1 | 1/0 | **1/-1** |
| Mamba SSM | 1/0 | 1/0 | 1/0 | 0/-1 | 0/-1 | 1/-1 |
| TabPFN | 1/0 | 1/0 | 1/0 | **2/0** | 1/0 | 0/-1 |

Column bests: No Reduction: Random forest, PCA: Random forest, KernelPCA: Random forest, Isomap: TabPFN, MDS: MLP, BOTCAST: RNN GRU

### Onion — best pairing: **KernelPCA + Random forest** (R2 = 0.5011)

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 1/-1 | 0/-1 | 0/-1 | 0/-2 | 0/-1 | 0/-1 |
| Random forest | 1/0 | 0/-1 | **2/0** | 1/0 | 1/0 | 1/-1 |
| k-NN | 1/0 | 1/0 | 1/0 | 1/0 | 1/0 | 0/-1 |
| MLP | 2/0 | 1/0 | 0/0 | 0/0 | 0/0 | 0/-1 |
| LSTM | **2/0** | 2/0 | 2/0 | 0/-1 | **2/0** | 1/-1 |
| RNN GRU | 1/-1 | **2/0** | 1/0 | 0/-1 | 2/0 | 1/-1 |
| Mamba SSM | 2/0 | 0/-1 | 1/0 | 0/-1 | 0/-1 | **2/-1** |
| TabPFN | 2/-1 | 2/0 | 0/-1 | **2/0** | 2/0 | 0/-1 |

Column bests: No Reduction: LSTM, PCA: RNN GRU, KernelPCA: Random forest, Isomap: TabPFN, MDS: LSTM, BOTCAST: Mamba SSM

**regression: 7 distinct winners across the 18 columns** — Random forest ×5, TabPFN ×5, RNN GRU ×3, LSTM ×2, k-NN ×1, MLP ×1, Mamba SSM ×1

## For comparison, the pooled tables

| Task | Pooled column bests | Distinct winners |
|---|---|---|
| classification | MLP ×3, TabPFN ×2, Random forest ×1 | 3 |
| regression | TabPFN ×3, Random forest ×2, Mamba SSM ×1 | 3 |
