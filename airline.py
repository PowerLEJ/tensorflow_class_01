# 파이썬 패키지 가져오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.layers import LSTM

# 하이퍼 파라미터
MY_PAST = 12 # 과거 몇 달치 사용할 건지
MY_SPLIT = 0.8 # 얼마만큼을 학습용으로 사용할지
MY_UNIT = 300 # LSTM셀 안에 내부 구조의 차원수
MY_SHAPE = (MY_PAST, 1) # 입력으로 들어갈 모양 (2차원)

MY_EPOCH = 300 # 반복 학습수
MY_BATCH = 64 # 병렬 계산 데이터수
np.set_printoptions(precision=3) # numpy내용 출력 시 소수점 3자리까지

############### 데이터 준비 #################

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
raw = pd.read_csv('airline.csv',
                  header=None,
                  usecols=[1])

# 시계열 데이터 시각화
# plt.plot(raw)
# plt.show()

# 데이터 원본 출력
print('원본 데이터 샘플 13개')
print(raw.head(13))

print('\n원본 데이터 통계')
print(raw.describe())

# MinMax 데이터 정규화
scaler = MinMaxScaler()
s_data = scaler.fit_transform(raw) # raw는 데이터프레임인데 numpy 형식으로 전환

print('\nMinMax 정규화 형식', type(s_data))

# 정규화 데이터 출력
df = pd.DataFrame(s_data) # s_data를 데이터프레임으로 역전환

print('\n정규화 데이터 샘플 13개')
print(df.head(13))

print('\n정규화 데이터 통계')
print(df.describe())

# 13개 묶음으로 데이터 분할
# 결과는 python 리스트
bundle = []
for i in range(len(s_data) - MY_PAST):
    bundle.append(s_data[i: i+MY_PAST+1])

# 데이터 분할 결과 확인
print('\n총 13개 묶음의 수:', len(bundle))
print(bundle[0])
print(bundle[1])

# numpy로 전환
print('분할 데이터의 타입:', type(bundle))
bundle = np.array(bundle)
print('분할 데이터의 모양:', bundle.shape)

# 데이터를 입력과 출력으로 분할
X_data = bundle[:, 0:MY_PAST] # 행은 처음부터 끝까지 즉 :, 열은 처음부터 11까지 즉 0:12
Y_data = bundle[:, -1] # 행은 처음부터 끝까지 즉 :, 열은 맨 마지막 즉 -1

# 데이터를 학습용과 평가용으로 분할
split = int(len(bundle) * MY_SPLIT) # 몇 프로를 학습용으로 할지
# 105개(13개 묶음)는 학습용, 27개는 평가용
X_train = X_data[: split]
X_test = X_data[split:]

Y_train = Y_data[: split]
Y_test = Y_data[split:]

# 최종 데이터 모양
print('\n학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)

print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)



################# 인공 신경망 구현 ################

# RNN 구현
# 케라스 RNN은 2차원 입력만 허용
model = Sequential()
model.add(InputLayer(input_shape=MY_SHAPE))
model.add(LSTM(MY_UNIT))

model.add(Dense(1,
                activation='sigmoid'))

print('\nRNN 요약')
model.summary()

################# 인공 신경망 학습 ###############

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='rmsprop',
              loss='mse') # 정확한 값을 들고 오지 않아도 된다

begin = time()
print('\nRNN 학습 시작')

model.fit(X_train,
          Y_train,
          epochs=MY_EPOCH,
          batch_size=MY_BATCH,
          verbose=0)

end = time()
print('총 학습 시간: {:.1f}초'.format(end - begin))

################## 인공 신경망 평가 #################

# RNN 평가
loss = model.evaluate(X_test,
                      Y_test,
                      verbose=1)

print('최종 MSE 손실값: {:.3f}'.format(loss))

# RNN 추측
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
pred = pred.flatten().astype(int)
print('\n추측 결과 원본: ', pred)

# 정답 역전환
truth = scaler.inverse_transform(Y_test)
truth = truth.flatten().astype(int)
print('\n정답 원본: ', truth)


# line plot 구성
axes = plt.gca()
axes.set_ylim([0, 650]) # y축의 최소값과 최대값의 범위 지정(0부터 650까지)

sns.lineplot(data=pred, label='pred', color='blue')
sns.lineplot(data=truth, label='truth', color='red')

plt.show()






