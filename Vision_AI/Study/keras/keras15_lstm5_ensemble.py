
from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], 
           [4,5,6], [5,6,7],[6,7,8], 
           [7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10,20,30], [20,30,40], [30,40,50], 
           [40,50,60], [50,60,70],[60,70,80], 
           [70,80,90],[80,90,100], [90,100,110], [100,110,120],
           [2,3,4], [3,4,5], [4,5,6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])




x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)  # (13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)  # (13, 3, 1)
print(x1.shape)
print(x2.shape)

# 2. 모델 구성
input1 = Input(shape=(3,1))
dense1 = LSTM(10, activation='relu')(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,1))
dense2_1 = LSTM(32, activation='relu')(input2)
dense2_2 = Dense(26)(dense2_1)
dense2_3 = Dense(8)(dense2_2)
output2 = Dense(3)(dense2_3)

from keras.layers.merge import concatenate, Add, Concatenate
# merge = concatenate([output1, output2])
# merge = Concatenate()([output1, output2])
merge = Add()([output1, output2])

output_1 = Dense(30)(merge) #1번째 output 모델
output_1 = Dense(30)(output_1) #Dense(n)을 col과 맞추기
output_1 = Dense(1)(output_1)

output_2 = Dense(20)(merge) #3번째 output 모델
output_2 = Dense(20)(output_2)
output_2= Dense(1)(output_2)

model = Model(inputs = [input1, input2], outputs=[output_1,output_2])
#model input이 2개이므로 리스트로 넣어줌


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

model.fit([x1,x2],[y1,y2], epochs=1000, batch_size=1,verbose=1, callbacks=[early_stopping])
# verbose 0: 진행 정도 안나옴
# verbose 1: default
# verbose n: 값이 커질 수록 생략 많이 됨

# 4. 평가 예측
loss = model.evaluate([x1,x2], [y1,y2], batch_size=1)
print("loss : ", loss)

x1_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]]) #(3, ) => (1, 3) => (1, 3, 1)
x2_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])
# x_input = array([50,60,70])
x1_input = x1_input.reshape(4, 3, 1)
x2_input = x2_input.reshape(4, 3, 1)

y_predict = model.predict([x1_input,x2_input])
print(y_predict)
