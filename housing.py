# 파이썬 패키지 가져오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터
MY_EPOCH = 500 # 학습 시 몇 번 반복할지
MY_BATCH = 64 # 한꺼번에 몇 개를 처리할지

############ 데이터 준비 #############

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
heading = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']

raw = pd.read_csv('housing.csv')

# 데이터 원본 출력
print('원본 데이터 샘플 10개')
print(raw.head(10))

print('원본 데이터 통계')
print(raw.describe())

# Z-점수 정규화
# 결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
z_data = scaler.fit_transform(raw)

# numpy에서 pandas로 전환 (numpy와 pandas 데이터 형식이 달라서)
# header 정보 복구 필요
z_data = pd.DataFrame(z_data, columns=heading)

# 정규화 된 데이터 출력
print('정규화 된 데이터 샘플 10개')
print(z_data.head(10))

print('정규화 된 데이터 통계')
print(z_data.describe())


# 데이터를 입력과 출력으로 분리
print('\n분리 전 데이터 모양: ', z_data.shape) # (506, 13) 506개의 데이터(엑셀에서 집값 데이터), 각각의 데이터가 13개로 이루어져 있다
X_data = z_data.drop('MEDV', axis=1) # axis=1 컬럼을 누락, axis=0 행을 누락
Y_data = z_data['MEDV']

# 데이터를 학습용과 평가용으로 분리
# X_train, X_test, Y_train, Y_test 순서 중요
# test_size는 평가용 데이터를 전체 데이터에서 무작위로 몇 퍼센트 무시할 건지 지정
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)

print('\n학습용 입력 데이터 모양', X_train.shape)
print('학습용 출력 데이터 모양', Y_train.shape)
print('평가용 입력 데이터 모양', X_test.shape)
print('평가용 출력 데이터 모양', Y_test.shape)

# 상자그림 출력
sns.set(font_scale=2)
sns.boxplot(data=z_data, palette='dark')
plt.show()


######## 인공 신경망 구현 ########

# 케라스 DNN 구현
model = Sequential()
input = X_train.shape[1] # (354, 12) 에서 1번째는 12이다
# 은닉층 1 Dense는 은닉층에 대한 뉴런 200개랑 입력층의 12개 뉴런을 곱해서 2400개 그리고 은닉층 뉴런 200개 더해서 총 2600개
model.add(Dense(200,
                input_dim=input,
                activation='relu'))
# 은닉층 2추가
model.add(Dense(1000,
                activation='relu'))
# 최종 출력층 추가
model.add(Dense(1))

print('\nDNN 요약')
model.summary()


######## 인공 신경망 학습 ########

# sgd : 함수의 기울기(경사)를 구하여 낮은 쪽으로 이동
# mse : (예측값 - 정답)제곱의 평균
# 최적화 함수와 손실 함수 지정 (인공신경망 학습 환경 설정)
model.compile(optimizer='sgd',
              loss='mse')

print('\nDNN 학습 시작')
begin = time()

# epochs 학습 시 몇 번 반복할지
# batch_size 한꺼번에 몇 개를 처리할지
# verbose 학습 내용의 결과를 매번 출력할지 0은 출력하지 않음, 1은 출력함
# < 인공신경망 학습 >
model.fit(X_train,
          Y_train,
          epochs=MY_EPOCH,
          batch_size=MY_BATCH,
          verbose=0
          )
end = time()
print('총 학습 시간: {:.1f}초'.format(end-begin))

######## 인공 신경망 평가 및 활용 ########

# 신경망 평가 및 손실값 계산
loss = model.evaluate(X_test,
                      Y_test,
                      verbose=1)
# 152개 문제를 한 번만 풀었다
print('\nDNN 평균 제곱 오차 (MSE): {:.2f}'.format(loss))

# 신경망 활용 및 산포도 출력
pred = model.predict(X_test)
sns.regplot(x=Y_test, y=pred) # x축은 정답, y축은 예측값

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

