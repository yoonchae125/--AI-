
#1. 데이터
import numpy as np

x1 = np.array([range(1,101),range(101,201),range(301,401)])
x2 = np.array([range(1001,1101),range(1101,1201),range(1301,1401)])
y = np.array([range(101, 201)])
print(x1.shape)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
  
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.4, shuffle=False)
# shuffle : default True
x1_test, x1_val,x2_test, x2_val, y_test, y_val = train_test_split(x1_test, x2_test, y_test, test_size=0.5, shuffle=False)


#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, Input

# 함수형 모델
input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense2_1 = Dense(5)(input2)
dense2_2 = Dense(2)(dense2_1)
dense2_3 = Dense(4)(dense2_2)
output2 = Dense(5)(dense2_3)

# 모델 병합
from keras.layers.merge import concatenate
merge = concatenate([output1, output2])

middle1 = Dense(4)(merge)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)


model = Model(inputs = [input1, input2], outputs = output)

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
model.fit([x1_train,x1_train],y_train, epochs=200, batch_size=1,validation_data=([x1_val,x2_val], y_val))
# ecpoch : 반복 횟수

# 4. 평가 예측
loss, mse = model.evaluate([x1_test,x2_test],y_test, batch_size=1)
# loss, mse = model.evaluate(x,y, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)


x1_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])
x2_pred = np.array([[501,502,503],[504,505,506],[507,508,509]])
x1_pred = np.transpose(x1_pred)
x2_pred = np.transpose(x2_pred)

aaa = model.predict([x1_pred,x2_pred], batch_size=1)
print(aaa)

y_predict = model.predict([x1_test,x2_test], batch_size=1)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ",RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
