import numpy as np
#1. 데이터 정제
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)
# print(y.shape)

#####################

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() 

model.add(Dense(5, input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
##훈련 / compile = 사람말을 기계에게 알려주기
# mse = 손실은 낮으면 좋다 여러가지가 있음
# optimizer = 통상적으로 80%으로 먹혀 들어감 대게 이걸 사용

'''
#3. model.fit = 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x,y, epochs = 1000, batch_size=1)

#4. 평가
loss, mse = model.evaluate(x,y, batch_size = 1)
print('mse: ' , mse)


x_prd = np.array([11,12,13])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)

bbb = model.predict(x, batch_size=1)
print(bbb)
'''


