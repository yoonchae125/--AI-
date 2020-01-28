
#1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=False)
# shuffle : default True
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)

print(x_train)
print(x_test)
print(x_val)
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
            metrics=['mse'])
# loss : 손실, 낮을수록 좋음
# mse(mean squared error) : 낮을수록 좋음
# optimizer : 최적화 - 보통 adam 사용
# metrics=['acc'] : 결과를 acc로 보여줌, 손실률(loss) 다음에 무엇을 보여줄 것인가?
# **회귀 문제에서는 'mae','mse' 사용, 'acc' 사용 안함!
model.fit(x_train,y_train, epochs=50, batch_size=1,validation_data=(x_val, y_val))
# ecpoch : 반복 횟수

# 4. 평가 예측
loss, mse = model.evaluate(x_test,y_test, batch_size=1)
# loss, mse = model.evaluate(x,y, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)

x_pred = np.array([101,102,103])
aaa = model.predict(x_pred, batch_size=1)
y_predict = model.predict(x_test, batch_size=1)
print(aaa)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ",RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)