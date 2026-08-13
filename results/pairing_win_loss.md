# Combined pairing wins/losses

Threshold: within 0.01 of the best/worst score; both the within-model DR comparison and within-DR model comparison are counted.

## classification

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 2/-3 | 0/-3 | 1/-3 | 0/-6 | 2/-4 | 3/-1 |
| Random forest | 5/-3 | 4/-2 | 2/-4 | 3/-3 | 4/-2 | 6/-1 |
| k-NN | 4/-3 | 4/-3 | 4/-2 | 3/-3 | 4/-3 | 2/-4 |
| MLP | 2/-3 | 5/-0 | 4/-3 | 2/-3 | 5/-2 | 4/-2 |
| LSTM | 2/-2 | 6/-1 | 2/-3 | 2/-2 | 4/-2 | 4/-2 |
| RNN GRU | 5/-3 | 6/-0 | 5/-0 | 5/-2 | 5/-1 | 0/-5 |

## regression

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 3/-0 | 1/-0 | 1/-0 | 0/-4 | 0/-2 | 2/-4 |
| Random forest | 5/-0 | 6/-0 | 4/-0 | 1/-1 | 5/-0 | 1/-2 |
| k-NN | 3/-0 | 2/-0 | 3/-0 | 3/-0 | 5/-0 | 1/-4 |
| MLP | 2/-0 | 2/-0 | 2/-0 | 1/-1 | 1/-0 | 1/-2 |
| LSTM | 1/-4 | 1/-3 | 1/-4 | 1/-1 | 2/-2 | 1/-3 |
| RNN GRU | 2/-1 | 2/-0 | 3/-1 | 4/-1 | 3/-0 | 3/-2 |
