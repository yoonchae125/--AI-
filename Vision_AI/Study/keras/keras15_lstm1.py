
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])
y = array([4,5,6,7,8])

print(x.shape) # (5, 3)
print(y.shape) # (5, )

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (5, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (열, 몇개씩 자르는지) 행 무시!
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=200, batch_size=1)
# ecpoch : 반복 횟수

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

x_input = array([6,7,8]) #(3, ) => (1, 3) => (1, 3, 1)
x_input = x_input.reshape(1, 3, 1)
y_predict = model.predict(x_input)
print(y_predict)