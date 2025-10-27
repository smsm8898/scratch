# Scratch
Understanding and Implementing the fundamental building blocks of Deep Learning from original papers

## Prerequisities
- python >= 3.9
- numpy >= 2.0.0
- scikit-learn >= 1.6.0

---
## 🔸 [Activation Functions](https://velog.io/@smsm8898/Study-Activation-Functions)

| No | Name                                  | PyTorch  | 특징                  | 
| -- | ------------------------------------- | -------------- | -------------------------- | 
| 1  | **Sigmoid**                           | `torch.nn.Sigmoid`   | 출력 0~1 / 이진 분류 시 사용        |
| 2  | **Tanh**                              | `torch.nn.Tanh`      | 출력 -1~1 / 중심화된 활성화         |
| 3  | **ReLU**                              | `torch.nn.ReLU`      | 양수만 통과 / sparse activation | 
| 4  | **Leaky ReLU**                        | `torch.nn.LeakyReLU` | 음수 영역도 작은 기울기 유지           | 
| 5  | **ELU (Exponential Linear Unit)**     | `torch.nn.ELU`       | 음수 영역 완만 / 평균 활성화 0 근처     |
| 6  | **GELU (Gaussian Error Linear Unit)** | `torch.nn.GELU`      | 확률적 / Transformer 계열에서 사용  |


---
## 🔸 [Optimizers Functions](https://velog.io/@smsm8898/Study-Optimizer-Functions)

| No | Name                                    | PyTorch        | 특징                | 
| -- | --------------------------------------- | -------------------------------- | ------------------------ | 
| 1  | **SGD**                                 | `torch.optim.SGD`                | 기본 확률적 경사 하강법            |
| 2  | **SGD + Momentum**                      | `torch.optim.SGD(momentum=0.9)`  | 모멘텀 적용 SGD               | 
| 3  | **Nesterov Accelerated Gradient (NAG)** | `torch.optim.SGD(nesterov=True)` | NAG / lookahead gradient | 
| 4  | **AdaGrad**                             | `torch.optim.Adagrad`            | 학습률 적응 / 과거 gradient 반영  |
| 5  | **RMSProp**                             | `torch.optim.RMSprop`            | 지수 이동 평균 기반 학습률 조절       |
| 6  | **Adam**                                | `torch.optim.Adam`               | Momentum + RMSProp 결합    | 
| 7  | **AdamW**                               | `torch.optim.AdamW`              | Weight Decay 분리 적용       |



## 🔸 Loss Functions
| No | Name                                  | PyTorch            | Type                         | 
| -- | ------------------------------------- | ------------------------ | ---------------------------- | 
| 1  | **Mean Squared Error Loss**           | `torch.nn.MSELoss`             | Regression                   |
| 2  | **Mean Absolute Error Loss / L1Loss** | `torch.nn.L1Loss`              | Regression                   |
| 3  | **Binary Cross Entropy Loss**         | `torch.nn.BCELoss`             | Binary Classification        |
| 4  | **Binary Cross Entropy with Logits**  | `torch.nn.BCEWithLogitsLoss`   | Binary Classification        |
| 5  | **Cross Entropy Loss**                | `torch.nn.CrossEntropyLoss`    | Multi-class Classification   |
| 6  | **Huber Loss / Smooth L1 Loss**       | `torch.nn.SmoothL1Loss`        | Regression / Robust          |
| 7  | **KL Divergence Loss**                | `torch.nn.KLDivLoss`           | Distribution / Probabilities |
| 8  | **Hinge Loss (Multi-margin)**         | `torch.nn.MultiMarginLoss`     | Classification               |
| 9  | **Cosine Embedding Loss**             | `torch.nn.CosineEmbeddingLoss` | Metric Learning              |
| 10 | **Triplet Margin Loss**               | `torch.nn.TripletMarginLoss`   | Metric Learning              |

