import numpy as np 

#1. 데이터 
x = np.array([range(1,101), range(101,201), range(301,401)])
y = np.array([range(101,201)])

# x= np.transpose(x)
x = x.reshape(x.shape[1], x.shape[0], 1)
y= np.transpose(y)
print(x.shape) #(3,100)
print(y.shape) #(1,100)


#2. 모델 불러오기
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = load_model('./save/savetest01.h5')
model.add(Dense(5,  name='dense2_1'))
model.add(Dense(7,  name='dense2_2'))
model.add(Dense(1,  name='dense2_3'))

model.summary()

#early stopping
from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph',
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True )

#early stopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') 

#3. 훈련- matrics: mse
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping, tb_hist]) 
    
#4. 평가 예측
loss, mae = model.evaluate(x, y, batch_size=1) #3.test
print('mae:' , mae)
print('loss:' , loss)


x_input = np.array([606,607,608]) # (3,) -> (1, 3) -> (1 , 3, 1) 전체를 곱한 값이 같기 때문에 reshape 가능
x_input = x_input.reshape(1,3,1)

y_predict = model.predict(x_input)
print(y_predict)
