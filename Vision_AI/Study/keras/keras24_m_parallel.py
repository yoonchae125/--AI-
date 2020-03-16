from numpy import array
import numpy as np

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix-1, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape) #(10,1)
print(out_seq.shape) #(10,1)

in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

#hstack
from numpy import hstack
dataset = hstack((in_seq1, in_seq2, out_seq)) #hstack: 10,1 3개가 합쳐져서 10,3
n_steps = 3
# print(dataset)

x, y = split_sequence2(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])


print(x.shape) # (7,3,3) 
print(y.shape) # (7,3)


  
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터

import numpy as np

x =  x.reshape(x.shape[0],x.shape[1]*x.shape[2])
# y= y.reshape(1,y.shape[0])

print(y.shape)


# print(x.shape) 
# print(y.shape) 
# x = x.reshape(x.shape[0], 1, n_steps)
# print(x.shape) 

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_shape=(6,)))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))
# model.add(LSTM(64, activation='relu', input_shape=(1,n_steps))) # (열, 몇개씩 자르는지) 행 무시!
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(1))

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=200, batch_size=1)
# ecpoch : 반복 횟수

# 4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

x_input = array([[90,95],[100,105],[110,115]])
x_input = x_input.reshape(1,6)
y_predict = model.predict(x_input)
print(y_predict)
