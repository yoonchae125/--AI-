# 딥러닝 주요 이론 & Tensorflow를 활용한 딥러닝 실습

### 1. 머신러닝 vs 딥러닝

### 2. 딥러닝 핵심 개념

### 3. 딥러닝 모델 최적화 이론

### 4. Tensorflow - Regression

### 5. Tensorflow - Classification

### 6. 분야별 딥러닝 활용 사례





## 1. 머신러닝 vs 딥러닝

**딥러닝**

<img src="image/image-20191227102917973.png" alt="image-20191227102917973" style="zoom:67%;" />

- DNN에 데이터 들어오고 채점 -> 파라미터 값 변경

- 데이터가 부족하면 잘 안됨, 과거에는 딥러닝 못했음

- 딥러닝이 가능하게 된 이유

  - many training data
  - changes in computing technology
  - progress in machine learning research

- 딥러닝은 Feature Engineering 직접 해주지 않아도 됨

  

  <img src="image/image-20191227103950170.png" alt="image-20191227103950170" style="zoom: 50%;" />

- ex) 손 X-ray 사진 보고 나이 맞추는 문제

  - 머신러닝

    > 뼈의 길이에 따라 나이를 예측하도록 모델을 훈련시킨다.

  - 딥러닝

    > 아주 많은 이미지를 모델에게 준다.
    
    > 모델의 이미지를 보고 아이의 나이를 예측하도록 학습한다.

  => 딥러닝이 사용하기 더 쉽다고 할 수 있음

  => 딥러닝은 세부적인 튜닝이 어려움

  => 분석 분야에 대한 지식이 있으면 머신러닝의 성능이 더 높음

- 딥러닝은 많은 양의 <u>정답이 달린 데이터</u>가 필요함

- 딥러닝은 블랙박스 모델임 : 파라미터의 가중치 알 수 없음

  > 최근에는 속성 값을 하나씩 가리면서 조금씩 알아가는 중

  > Explainable AI (설명 가능 인공 지능)

  




## 2. 딥러닝 핵심 개념

**Single-Layer Perceptron**(단층 퍼셉트론)

- 뉴런을 본따 만든 알고리즘 하나의 단위

- 2가지 연산을 적용하여 출력값 계산

  - Linear combination + <u>Activation function</u>

    > x1*(w11+w12)+x2*(w11+w12)+(bias1+bias2)

  - **Activation Functions**

    > 아무리 Layer를 쌓아도(Linear combination) Linear Regression (y=ax+b)

    >이전 레이어의 모든 입력에 대한 가중합을 받아 출력 값을 생성하여 다음 레이어로 전달하는 <u>비선형 함수</u>

    <img src="image/image-20191227140119212.png" alt="image-20191227140119212" style="zoom: 67%;" />

    > Step Function은 사용 못함 : 미분했을 때 값이 0이기 때문 => Sigmoid 함수 사용

    > but **Sigmoid** 미분 max : 0.25
    >
    > - 미분 여러번 하면 값이 매우 작아져서 MSE가 사라져버림 : Vanishing Gradient

    <img src="image/image-20191227140835726.png" alt="image-20191227140835726" style="zoom:80%;" />

    

    > **ReLU** : 미분하면 0 or 1

    > **Leaky ReLU** : ReLU 함수 0으로 미분 되는 부분 살리기 위해 발전시킨 함수

- AND, OR 문제 해결 가능, XOR 해결 불가능ㅁ



**Multi-Layer Perceptron (MLP, 다층 퍼센트론)**

- 복수의 Perceptron 을 연결한 구조

<img src="image/image-20191227144523574.png" alt="image-20191227144523574" style="zoom:50%;" />

- XOR 문제 해결 가능
- Layer 깊어지면 overfitting 일어날 수 있음





**Artificial Neural Network (인공 신경망)**

- Perceptron을 모은 Layer를 깊이 방향으로 쌓아나가면서 복잡한 모델을 만들어내어 보다 더 어려운 문제를 풀어낼 수 있음
- MLP도 ANN의 일종

<img src="image/image-20191227145430835.png" alt="image-20191227145430835" style="zoom:67%;" />

- Input Layer

  > 외부로 부터 데이터를 입력 받는 신경망의 입구 layer

- Hidden Layer

  > Input layer와 Output layer 사이의 모든 layers
  >
  > == Learnable Kernels

- Output Layer

  > 모델의 최종 연산 결과를 내보내는 신경망 출구의 layer

  > 결과 값을 그대로 받아 Regression

  > Sigmoid를 거쳐 Binary Classification

  > Softmax를 거쳐 K-Class Classification



**Back Propagation Algorithm (오차 역전파 알고리즘)**

![image-20191227150303983](image/image-20191227150303983.png)

-  신경망의 효율적인 학습 방법
- 학습된 출력 값과 실제 값과의 차이인 오차를 계산하여 Feedforward 반대인 역방향으로 전파하는 알고리즘
- ex) MLP로 XOR해결
  - Layer 복잡해지면 연산이 복잡해짐->비효율적
  - 이를 해결하기 위해 Back Propagation 도입
  - Forward방향으로 한번 연산을 한 다음 그 결과값(Cost)을 역방향(Backward)으로 전달해가면서 Parameter를 Update (cost 최소인 parameter구함)
- [Neural Networks Test](http://playground.tensorflow.org/)





## 3. 딥러닝 모델 최적화 이론

**1. Weight Initialization**

- Gradient descent를 적용하기 위한 첫단계는 모든 Parameter를 초기화하는 것

- 초기화 시점의 작은 차이가 학습의 결과를 바꿀 수 있음

- Linear combination 결과값이 너무 커지거나 작아지지 않게 만들어주는 것이 중요

- 발전된 초기화 방법들을 활용해 Vanishing gradient 혹은 Exploding gradient 문제를 줄일 수 있음

  > **Xavier Initailization**
  >
  > - 활성화 함수로 Sigmoid나 tanh를 사용할 때 적용
  > - 다수의 딥러닝 라이브러리들에 Default로 적용되어 있음
  > - 표준편차가 `√1/n` 인 정규분포를 따르도록 가중치 초기화
  >
  > **He Initialization**
  >
  > - 활성화 함수가 ReLU일 때 적용
  > - 표준편차가 `√2/n` 인 정규분포를 따르도록 가중치 초기화

**2. Weight regularization**

- 기존의 Gradient Descent 계산 시 y축에 위치해 있던 Cost function은 training data에 의해 모델이 발생시키는 Error값의 지표 

- training data만 고려된 cost function을 기준으로 gradient descent를 적용하면 <u>overfitting</u>에 빠질 수 있음

- 모델이 복잡해지면 θ의 갯수 많아지고 절대값이 커지는 경향이 있음

- 값이 커지는 θ에 대한 함수를 기존의 Cost function에 더해 Trade-off관계 속에서 최적값을 찾을 수 있음

  <img src="image/image-20191227160703758.png" alt="image-20191227160703758" style="zoom:50%;" />

  > **L1 regularization (L1 정규화) : Ridge**
  >
  > - 가중치의 절대값의 합에 비례하여 가중치에 페널티를 줌
  > - 관련성이 없거나 매우 낮은 특성의 가중치를 정확히 0으로 유도해 모델에서 해당 특성을 배제하는 효과 (==Feature selection)
  >
  > **L2 regularization (L2 정규화) : Lasso**
  >
  > - 가중치의 제곱의 합에 비례하여 가중치에 페널티를 줌
  > - 큰 값을 가진 가중치를 더욱 제약하는 효과
  >
  > **Regularization Rage (λ) : 정규화율**
  >
  > - 스칼라 값
  > - 정규화 함수의 상대적 중요도를 지정
  > - 정규화율을 높이면 과적합이 감소되지만 모델의 정확성이 떨어질 수 있음
  > - θ의 수가 아주 많은 신경망은 정규화율을 아주 작게 주기도 함

**3. Advanced gradient descent algorithms**

> **(Full-Batch) Gradient Descent**
>
> <img src="image/image-20191227164603155.png" alt="image-20191227164603155" style="zoom:50%;" />
>
>  - 모든 training data에 대해 cost를 구하고 cost function값을 구한 다음 이를 기반으로 fraient descent 적용
>
>  - training data가 많으면 cost function 등의 계산에 필요한 연산의 양이 많아진다. 
>
>    => 학습이 오래 걸림
>
> - Weight initialization 결과에 따라 global minimum이 아닌 local minimum으로 수렴할 수 있다

> **Stochastic Gradient Descent (SGD, 확률적 경사 하강법)**
>
> <img src="image/image-20191227164711860.png" alt="image-20191227164711860" style="zoom:50%;" />
>
> - 하나의 training data마다 cost계산(batch size=1)하고 바로 gradient descent적용해 weight 빠르게 update
> - 신경망의 성능이 들쑥날쑥 변함 (Cost값이 안정적으로 죽어들지 않음)
> - 최적의 learning rate를 구하기 위해 일일이 튜닝하고 수렴조건(early-stop)을 조정해야 함

> **Mini-Batch Stochastic Gradient Descent**
>
> ![image-20191227165116186](image/image-20191227165116186.png)
>
> - training data에서 일정한 크기의 데이터를 선택(batch size)하여 cost function 계산 및 gradient descent적용
> - 앞선 두가지 기법의 단점 보완하고 장점을 취함
> - 설계자의 의도에 따라 속도와 안정성을 동시에 관리할 수 있으며, GPU기반의 효율적이 병렬 연산이 가능

- Epoch : 전체 학습 데이터를 한 번씩 모두 학습시킨 횟수

![백날 자습해도 이해 안 가던 딥러닝, 머리 속에 인스톨 시켜드립니다 by 하용호](image/image-20191227165812228.png)

> **Adam (Adaptive Moment Estimation) Optimizer**
>
> - Momentum과 AdaGrad/RMSProp의 이점 조합
> - Adaptive learning rate가 적용되어 learning rate에 대한 탐색의 필요성이 줄어듦



**Avoiding Overfitting**

1. Droupout

   <img src="image/image-20191227171025815.png" alt="image-20191227171025815" style="zoom:67%;" />

   - 신경망에 적용할 수 있는 효율적인 overfitting 회피 방법 중 하나

   - training 진행할 때 매 batch마다 layer 단위로 일정 비율 만큼의 neuron을 꺼뜨리는 방식

   - Test / Inference 단계에는 dropout을 걷어내어 전체 neuron이 살아 있는 채로 inference를 진행해야 함

   - 랜덤하게 neuron을 꺼뜨려 학습을 방해하여 모델을 학습이 training data에 편향되는 것을 막아줌

   - 동일한 데이터에 대한 매번 다른 모델을 학습시키는 것과 마찬가지의 효과 발생 

     => Model ensemble 효과

   - 가중치 값이 큰 특정 neuron의 영향력이 커져 다른 neuron들의 학습 속도에 문제를 발생시키는 Co-adaptation을 회피할 수 있게 함

   

2. Batch Normalization

   - input data에 대해 Standardization과 같은 Normalization을 적용하면 전반적으로 model의 성능이 높아짐

     => 데이터 내 Column들의 scale에 model이 너무 민감해지는 것을 막아주기 때문

     ![image-20191227171605931](image/image-20191227171605931.png)

   - 신경망의 경우 normalization이 제대로 적용되지 않으면 최적의 cost지점으로 가는 길을 빠르게 찾지 못함

   - input data 뿐만 아니라 신경망 내부의 hidden layer의 input에도 적용해 주는 것이 BN

   - activation function적용 전에 BN 적용

   - BN 과정

     > 1. hidden layer로의 input data에 대해 평균이 0, 분산이 1이 되도록 normalization 진행
     > 2. hidden layer의 output이 비선형성을 유지할 수 있도로 normalization 결과에 Scaling(`*r`) & Shifting(`+ß`) 적용
     > 3. Scaling&Shifting을 적용한 결과를 activation function에 전달 후 hidden layer의 최종 output 계산

   - 장점

     - 학습 속도 & 학습 결과 개선
     - 가중치 초기값에 크게 의존하지 않음
     - overfitting 억제







## 4. Tensorflow - Regression

**Tensorflow 특징**

- **Python API **(C, C++, Go, Java, Javascript, Swift, ...)
- **Portability** : deploy computation to one or more CPUs or GPUs in a desktop, server, mobile device with a single API
- **Flexibility** : from Rasberry Pi, Android, Windows, iOS, Linux to server farms
- **Visualization & Event logging** : user TensorBoard
- **Checkpoints** : for managing experiments, use tf.Saver
- Auto-differentiation autodiff
- Large community & Awesome projects already using Tensorflow



**Tensorflow basic**

- 계산 두 단계

  1. Building a Tensorflow Graphs

  ```python
  a = tf.add(3, 5)
  ```

  

  2. Executing the Tensorflow Graph

  ```python
  sess = tf.Session() # == thread
  print(sess.run(a))
  sess.close() #닫아줌
  
  #위 코드와 동일
  with tf.Session() as sess:
    print(sess.run(a))
  ```



**Regression with Neural Network**

![image-20191230135232461](/Users/chaeyoon/Documents/GitHub/AI_Image/part5/image/image-20191230135232461.png)

- Network + Cost Function + Optimizer



**Placeholders**

- 계산 그래프를 실행할 때 사용자가 실제 데이터를 흘려보낼 수 있는 통로

## 5. Tensorflow - Classification

### 



## 6. 분야별 딥러닝 활용 사례

