
#1. 데이터
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

# 3.훈련
model.compile(loss='mse', optimizer='adam',
            #    metrics=['acc'])
            metrics=['mae'])
# loss : 손실, 낮을수록 좋음
# mse(mean squared error) : 낮을수록 좋음
# optimizer : 최적화 - 보통 adam 사용
# metrics=['acc'] : 결과를 acc로 보여줌, 손실률(loss) 다음에 무엇을 보여줄 것인가?
# **회귀 문제에서는 'mae','mse' 사용, 'acc' 사용 안함!
model.fit(x,y, epochs=100, batch_size=1)
# ecpoch : 반복 횟수

# 4. 평가 예측
# loss, acc = model.evaluate(x,y, batch_size=1)
loss, mae = model.evaluate(x,y, batch_size=1)
# print('acc: ', acc)
print('acc: ', mae)

x_pred = np.array([11,12,13])
aaa = model.predict(x_pred, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)