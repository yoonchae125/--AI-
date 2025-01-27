#1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
y_train = y[:60]
x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:]
y_val = y[80:]

# print(x.shape)
# print(y.shape)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=1))
model.add(Dense(5, input_shape=(1, )))

model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


# 3.훈련
model.compile(loss='mse', optimizer='adam',
            #    metrics=['acc'])
            metrics=['acc'])
# loss : 손실, 낮을수록 좋음
# mse(mean squared error) : 낮을수록 좋음
# optimizer : 최적화 - 보통 adam 사용
# metrics=['acc'] : 결과를 acc로 보여줌, 손실률(loss) 다음에 무엇을 보여줄 것인가?
# **회귀 문제에서는 'mae','mse' 사용, 'acc' 사용 안함!
model.fit(x_train,y_train, epochs=200, batch_size=1,validation_data=(x_val, y_val))
# ecpoch : 반복 횟수

# 4. 평가 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=1)
# loss, mse = model.evaluate(x,y, batch_size=1)
print('loss: ', loss)
print('acc: ', acc)

x_pred = np.array([101,102,103])
aaa = model.predict(x_pred, batch_size=1)
print(aaa)
