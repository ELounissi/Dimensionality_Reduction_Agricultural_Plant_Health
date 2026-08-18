# All-models comparison: rankings and win/loss tables

## classification

Overall means: TabPFN 0.8057  >  MLP 0.7949  >  LSTM 0.7931  >  Random forest 0.7923  >  RNN GRU 0.7899  >  k-NN 0.7887  >  Mamba SSM 0.7876  >  Decision tree 0.7798

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 1/-2 | 1/-3 | 1/-4 | 0/-5 | 1/-3 | 3/-2 |
| Random forest | 2/-2 | 1/-3 | 1/-3 | 2/-1 | 4/0 | **5/-1** |
| k-NN | 4/-4 | 4/-3 | 2/-4 | 4/-2 | 4/-4 | 2/-4 |
| MLP | **5/-1** | **5/0** | 4/-1 | 3/-3 | **5/0** | 2/-1 |
| LSTM | 2/-4 | 5/-1 | 4/-2 | 3/-3 | 4/-2 | 3/-2 |
| RNN GRU | 3/-2 | 5/-1 | 3/-1 | 2/-3 | 5/0 | 3/-3 |
| Mamba SSM | 2/-2 | 5/-1 | 4/-3 | 2/-1 | 2/-4 | 4/-3 |
| TabPFN | 5/-2 | 5/-2 | **5/-2** | **5/-2** | 4/-2 | 5/-2 |

Column bests: No Reduction: MLP, PCA: MLP, KernelPCA: TabPFN, Isomap: TabPFN, MDS: MLP, BOTCAST: Random forest

## regression

Overall means: Random forest 0.4820  >  TabPFN 0.4804  >  RNN GRU 0.4687  >  Mamba SSM 0.4657  >  LSTM 0.4650  >  MLP 0.4648  >  k-NN 0.4514  >  Decision tree 0.4055

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 2/-2 | 1/-2 | 0/-3 | 0/-4 | 0/-3 | 1/-2 |
| Random forest | **5/0** | 3/-1 | **4/-1** | 2/-1 | 3/0 | 1/-2 |
| k-NN | 3/0 | 2/0 | 2/0 | 4/0 | 4/0 | 0/-4 |
| MLP | 3/0 | 3/0 | 1/0 | 0/0 | 2/-1 | 1/-2 |
| LSTM | 4/0 | 3/0 | 3/0 | 1/-3 | 4/0 | 2/-3 |
| RNN GRU | 2/-3 | 3/-1 | 3/0 | 0/-3 | 3/0 | 2/-2 |
| Mamba SSM | 4/0 | 1/-1 | 2/-1 | 0/-3 | 0/-2 | **3/-2** |
| TabPFN | 5/-2 | **5/-1** | 2/-2 | **5/-1** | **5/-1** | 2/-2 |

Column bests: No Reduction: Random forest, PCA: TabPFN, KernelPCA: Random forest, Isomap: TabPFN, MDS: TabPFN, BOTCAST: Mamba SSM

