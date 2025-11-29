# Scratch
- Study and implement the parts I don’t understand while reproducing the paper
- TODO
  - DataLoader 작동원리
    - num_workers
    - pin_memory
    - persistent_workers
  - Training
    - gradient_clip
    - gradient_clip_algorithm
    mixed precision(amp)



## Prerequisities
- python >= 3.9
- numpy >= 2.0.0
- scikit-learn >= 1.6.0

## Experiment
- To check the functions, open `activations`, `optimizers`, `losses`
- The experiment is implemented on `notebooks`
  - For `Augmentation` experiment, I used `torchvision.transforms` functions
  - For `learning rate` experiment, I user `torch.optim`, `torch.optim.lr_scheduler`, and customize for `warmup`.


---
<br>

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
<br>

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


---
<br>

## 🔸 [Loss Functions](https://velog.io/@smsm8898/Study-Loss-Functions)

| No | Name                                  | PyTorch            | Type                         | 
| -- | ------------------------------------- | ------------------------ | ---------------------------- | 
| 1  | **Mean Squared Error Loss**           | `torch.nn.MSELoss`             | Regression                   |
| 2  | **Mean Absolute Error Loss / L1Loss** | `torch.nn.L1Loss`              | Regression                   |
| 3  | **Huber Loss / Smooth L1 Loss**       | `torch.nn.SmoothL1Loss`        | Regression / Robust          |
| 4  | **Binary Cross Entropy Loss**         | `torch.nn.BCELoss`             | Binary Classification        |
| 5  | **Cross Entropy Loss**                | `torch.nn.CrossEntropyLoss`    | Multi-class Classification   |


---
<br>

## 🔸 [Augmentation](https://velog.io/@smsm8898/Study-Augmentationvision)

| No | Augmentation | 정의 | 동작 원리 | 학습 효과 |
|----|--------------|------|-----------|-----------|
| 1  | **RandomHorizontalFlip** | 이미지를 좌우로 뒤집는 증강 | 확률 `p`에 따라 좌우 반전 | 좌우 대칭 객체나 방향 민감도 감소, 강건한 학습 |
| 2  | **RandomResizedCrop / CenterCrop** | 이미지 일부 영역 선택 후 지정 크기로 조정 | 랜덤/중앙 위치 선택 → crop → resize | 다양한 위치 학습, 위치 변화에 강건 |
| 3  | **ColorJitter** | 밝기, 대비, 채도, 색조 랜덤 변환 | 지정 범위 내 요소 무작위 조정 | 조명/색상 변화에 강건한 학습 |
| 4  | **RandomRotation** | 이미지 지정 각도 범위 내 회전 | 랜덤 각도 선택 → 이미지 회전, 필요시 padding | 방향 변화에 강건, 회전 불변 특징 학습 |
| 5  | **RandomAffine** | 회전, 이동, 스케일, 전단 등 복합 변형 | 지정 범위 내 변형 적용 | 위치, 크기, 형태 변화에 강건 |
| 6  | **RandomGrayscale** | 일정 확률로 이미지 흑백 변환 | 확률 `p`로 변환 → 3채널 유지 | 색상 의존도 감소, 구조적 특징 학습 강화 |
| 7  | **GaussianBlur** | 가우시안 필터로 흐림 처리 | 커널 크기 & 표준편차 → 이미지 convolution | 디테일 변화, 노이즈 강건성 향상 |
| 8  | **RandomErasing / Cutout** | 이미지 일부 영역 지우기/가림 | 랜덤 위치 & 크기 선택 → 픽셀 0 또는 지정값 채움 | Occlusion 대응, 일부 정보 손실에도 강건 |
| 9  | **Mixup** | 두 이미지 + 라벨을 가중 평균하여 합성 | λ 샘플링 → `x_new = λ*x1 + (1-λ)*x2`, `y_new = λ*y1 + (1-λ)*y2` | 경계 민감도 감소, 일반화 성능 향상, 오버피팅 감소 |
| 10 | **CutMix** | 이미지 일부 영역 교체 + 라벨 혼합 | x2 일부 영역 x1에 덮어쓰기 → λ = 패치 비율 → y_new 혼합 | Occlusion 대응, Mixup보다 현실적 혼합, 일반화 성능 향상 |


---
<br>

## 🔸 [Scheduler](https://velog.io/@smsm8898/Study-Learning-Scheduler)

| No | Scheduler             | 정의                                       | 동작 원리                                              | 학습 효과                                                     |
| -- | --------------------- | ---------------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| 1  | **StepLR**            | 일정 epoch 간격마다 lr을 감소시키는 스케줄러             | `step_size` epoch마다 `lr = lr * gamma`              | 안정적인 수렴, 간단하고 기본적인 lr 스케줄링                                |
| 2  | **CosineAnnealingLR** | lr을 cosine 곡선 형태로 부드럽게 감소                | `T_max` epoch 동안 lr이 최대 → 최소(`eta_min`)로 감소        | 부드러운 lr 감소로 모델이 세밀하게 수렴, 일반화 성능 향상                        |
| 3  | **ReduceLROnPlateau** | val loss가 개선되지 않을 때 lr 감소                | `patience` epoch 동안 개선 없으면 `lr = lr * factor`      | 자동 lr 조정으로 불필요한 감소 방지, fine-tuning과 transfer learning에 유리 |
| 4  | **OneCycleLR**        | lr을 한 주기 동안 올렸다가 내리는 방식, momentum 반대로 조절 | 초기 lr → `max_lr`로 증가 → 다시 감소, momentum은 lr과 반대로 변함 | 빠른 탐색과 안정적 수렴, 일반화 성능 향상, 최신 CNN/Transformer 학습에서 자주 사용   |


---
<br>

## 🔸 [Tokenizer](https://velog.io/@smsm8898/Study-Tokenizersnlp)
| No | Tokenizer                    | 정의                                                  | 동작 원리                                                                                   | 학습 효과                                                           |
| -- | ---------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1  | **BPE (Byte-Pair Encoding)** | 가장 자주 등장하는 문자쌍(bigram)을 반복적으로 병합해 서브워드 생성           | ① 말뭉치를 문자 단위로 분리 → ② bigram 빈도 계산 → ③ 가장 많이 등장한 pair 병합 → ④ 반복 수행                       | 희귀 단어 분해 능력이 높고, 안정적이며 빠른 학습 / 하지만 너무 자주 등장하는 패턴에 과도하게 병합될 수 있음 |
| 2  | **WordPiece**                | 확률 최대화(Likelihood maximization) 기반 병합을 수행하는 서브워드 모델 | ① 후보 서브워드를 만들고 점수(score)를 계산 → ② corpus likelihood가 가장 커지는 병합 선택 → ③ 반복                 | BPE보다 더 언어 모델링 목표에 맞춘 병합 생성 → 희귀 단어 처리 개선 / 다만 학습 속도는 더 느림      |
| 3  | **Unigram**                  | 단일 서브워드 집합 중 최적의 조합을 선택하는 확률 모델                     | ① 충분히 큰 서브워드 후보 집합 생성 → ② 각 서브워드에 확률 부여 → ③ 전체 likelihood를 가장 높이는 방향으로 서브워드 제거(pruning) | 가장 유연하고 자연스럽게 분해되는 토큰 집합을 만들며, 한국어·일본어 같은 형태소 복잡한 언어에서 탁월함      |

---
<br>

## 🔸 Normalization
| No | 이름                                        | 적용 대상                 | 정의                                    | 장점                                     | 단점                                 |
| -- | ----------------------------------------- | --------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------- |
| 1  | **Batch Normalization (BatchNorm)**       | Batch 방향              | 같은 채널에 대해 **batch 전체 평균·분산**을 사용해 정규화 | CNN에서 매우 효과적, 학습 안정·빠름                 | batch 크기에 민감, RNN·Transformer에 부적합 |
| 2  | **Layer Normalization (LayerNorm)**       | Feature 방향            | 한 샘플의 **모든 feature 평균·분산**으로 정규화      | Transformer, RNN에서 필수, batch 크기 영향 없음  | CNN에서는 효과 떨어짐                      |
| 3  | **Instance Normalization (InstanceNorm)** | 이미지 채널별               | 한 이미지의 **채널별 평균·분산**을 사용              | Style Transfer에서 매우 강함                 | 일반 분류 모델에는 약함                      |
| 4  | **Group Normalization (GroupNorm)**       | Feature 그룹            | 채널을 **여러 그룹으로 나눠 그룹별 정규화**            | 작은 batch에서도 안정적, CNN에서 BatchNorm 대체 가능 | Group 수 선택이 중요                     |