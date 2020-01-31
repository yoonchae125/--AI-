
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], 
           [4,5,6], [5,6,7],[6,7,8], 
           [7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) # (13, 3)
print(y.shape) # (13, )


x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (13, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(3,1))) # (열, 몇개씩 자르는지) 행 무시!
# model.add(LSTM(64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
# fit가 중지 됨
# patience: 20번 참고 학습 멈춤
# patience가 길어지면 과적합 구간에 걸려 더 안좋아 지는 경우가 있음
# early_stopping = EarlyStopping(monitor='acc', patience=40, mode='max')
# acc(monitor) 최대값을 찾았어, 그 다음에 최대값보다 큰걸 못찾아, 40번(patience) 참고 멈춤

model.fit(x,y, epochs=1000, batch_size=1,verbose=2, callbacks=[early_stopping])
# verbose 0: 진행 정도 안나옴
# verbose 1: default
# verbose n: 값이 커질 수록 생략 많이 됨

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90]]) #(3, ) => (1, 3) => (1, 3, 1)
# x_input = array([50,60,70])
x_input = x_input.reshape(3, 3, 1)
y_predict = model.predict(x_input)
print(y_predict)
