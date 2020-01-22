import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)

#####################

# 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() 

model.add(Dense(128, input_dim = 1))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(18))
model.add(Dense(9))
model.add(Dense(1))

##훈련 / compile = 사람말을 기계에게 알려주기
# mse = 손실은 낮으면 좋다 여러가지가 있음
# optimizer = 통상적으로 80%으로 먹혀 들어감 대게 이걸 사용
# model.fit = 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x,y, epochs = 1000, batch_size=1)

loss, mse = model.evaluate(x,y, batch_size = 1)
print('mse: ' , mse)


x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)



