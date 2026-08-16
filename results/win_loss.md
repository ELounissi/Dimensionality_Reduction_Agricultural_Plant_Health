# All-models comparison: rankings and win/loss tables

## classification

Overall means: TabPFN 0.8057  >  RNN GRU 0.8028  >  LSTM 0.8001  >  MLP 0.7994  >  Random forest 0.7953  >  k-NN 0.7939  >  Mamba SSM 0.7920  >  Decision tree 0.7805

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 2/-3 | 0/-2 | 0/-3 | 0/-6 | 2/-4 | 3/-2 |
| Random forest | 1/-4 | 4/-4 | 4/-3 | 4/-2 | 3/-1 | 4/-1 |
| k-NN | 4/-3 | 4/-3 | 4/-3 | 5/-2 | 4/-4 | 2/-5 |
| MLP | 3/-1 | **6/0** | 3/-1 | 2/-2 | **6/-1** | 2/-2 |
| LSTM | 5/-3 | 5/0 | 4/-3 | 3/-4 | 0/-4 | 3/-2 |
| RNN GRU | 6/-3 | 6/-3 | 3/-2 | 2/-2 | 2/-3 | **4/-1** |
| Mamba SSM | 4/-2 | 4/-1 | 4/-3 | 2/-3 | 2/-3 | 4/-1 |
| TabPFN | **5/-1** | 4/-2 | **5/-1** | **5/-2** | 3/-2 | 4/-2 |

Column bests: No Reduction: TabPFN, PCA: MLP, KernelPCA: TabPFN, Isomap: TabPFN, MDS: MLP, BOTCAST: RNN GRU

## regression

Overall means: Random forest 0.4694  >  TabPFN 0.4665  >  RNN GRU 0.4605  >  LSTM 0.4546  >  k-NN 0.4449  >  Mamba SSM 0.4441  >  MLP 0.4365  >  Decision tree 0.4045

| Model | No Reduction | PCA | KernelPCA | Isomap | MDS | BOTCAST |
|---|---|---|---|---|---|---|
| Decision tree | 2/-1 | 2/-2 | 0/-3 | 0/-4 | 0/-2 | 1/-4 |
| Random forest | 3/0 | 3/-1 | **4/0** | 3/-1 | 3/0 | 2/-2 |
| k-NN | **4/-1** | 0/-1 | 2/0 | 1/0 | 3/0 | 0/-4 |
| MLP | 2/0 | 2/-1 | 1/0 | 1/-1 | 2/0 | 1/-2 |
| LSTM | 3/-1 | **4/0** | 1/-2 | 1/-1 | **4/0** | 1/-2 |
| RNN GRU | 3/-1 | 4/0 | 2/0 | 2/-1 | 3/0 | 2/-2 |
| Mamba SSM | 1/-2 | 0/-3 | 1/-1 | 1/-3 | 1/-2 | **3/-2** |
| TabPFN | 2/-1 | 2/0 | 1/-2 | **5/-1** | 2/-1 | 2/-2 |

Column bests: No Reduction: k-NN, PCA: LSTM, KernelPCA: Random forest, Isomap: TabPFN, MDS: LSTM, BOTCAST: Mamba SSM

