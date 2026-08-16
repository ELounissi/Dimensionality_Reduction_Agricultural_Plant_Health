# Wilcoxon signed-rank tests

Seeds are averaged before testing. Raw p-values are shown in the readable tables. Holm-adjusted p-values are saved in statistical_tests.csv.

## Dimensionality reduction against No Reduction

| Task | PCA | MDS | Isomap | KernelPCA | BOTCAST |
|---|---|---|---|---|---|
| classification | 0.1297 (+0.0087) | 0.2837 (+0.0071) | 0.0008* (+0.0042) | 0.0737 (+0.0041) | 0.7337 (-0.0241) |
| regression | 0.7019 (-0.0108) | 0.2837 (-0.0124) | 0.0047* (-0.0300) | 0.0987 (-0.0015) | 0.0120* (-0.0667) |

## DR pairwise comparison

| | No Reduction | PCA | MDS | Isomap | KernelPCA | BOTCAST |
|---|---|---|---|---|---|---|
| No Reduction | --- | 0.6922 (-) | 0.7386 (+) | 0.0000* (+) | 0.0127* (+) | 0.0127* (+) |
| PCA | 0.6922 (+) | --- | 0.5813 (+) | 0.0000* (+) | 0.0219* (+) | 0.0041* (+) |
| MDS | 0.7386 (-) | 0.5813 (-) | --- | 0.0000* (+) | 0.1152 (+) | 0.0078* (+) |
| Isomap | 0.0000* (-) | 0.0000* (-) | 0.0000* (-) | --- | 0.0000* (-) | 0.4891 (+) |
| KernelPCA | 0.0127* (-) | 0.0219* (-) | 0.1152 (-) | 0.0000* (+) | --- | 0.0854 (+) |
| BOTCAST | 0.0127* (-) | 0.0041* (-) | 0.0078* (-) | 0.4891 (-) | 0.0854 (-) | --- |

## Model pairwise comparison

| | Decision tree | Random forest | k-NN | MLP | LSTM | RNN GRU |
|---|---|---|---|---|---|---|
| Decision tree | --- | 0.0000* (-) | 0.0003* (-) | 0.0056* (-) | 0.3677 (+) | 0.0000* (-) |
| Random forest | 0.0000* (+) | --- | 0.0201* (+) | 0.0065* (+) | 0.0002* (+) | 0.3624 (+) |
| k-NN | 0.0003* (+) | 0.0201* (-) | --- | 0.1568 (+) | 0.0023* (+) | 0.1227 (+) |
| MLP | 0.0056* (+) | 0.0065* (-) | 0.1568 (-) | --- | 0.0000* (+) | 0.0074* (-) |
| LSTM | 0.3677 (-) | 0.0002* (-) | 0.0023* (-) | 0.0000* (-) | --- | 0.0000* (-) |
| RNN GRU | 0.0000* (+) | 0.3624 (-) | 0.1227 (-) | 0.0074* (+) | 0.0000* (+) | --- |
