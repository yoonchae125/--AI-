from numpy import array

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)
dataset = [10,20,30,40,50,60,70,80,90,100]
n_steps = 3

x, y = split_sequence(dataset, n_steps)

# DNN 모델 구성

# loss 출력

# 90, 100, 110 예측


from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터


print(x.shape) 
print(y.shape) 


x = x.reshape(x.shape[0], 1, n_steps)
print(x.shape) 

# 2. 모델 구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(1,n_steps))) # (열, 몇개씩 자르는지) 행 무시!
model.add(Dense(32))
model.add(Dense(32))
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

x_input = array([90, 100, 110]) #(3, ) => (1, 3) => (1, 3, 1)
x_input = x_input.reshape(1, 1, 3)
y_predict = model.predict(x_input)
print(y_predict)
