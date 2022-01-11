# 파이썬 패키지 가져오기
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
import glob

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential

# 하이퍼 파라미터
MY_GEN = 128 # 생성자에 들어 있는 은닉층 각각의 뉴런의 숫자
MY_DIS = 128 # 감별자에 들어 있는 은닉층 각각의 뉴런의 숫자
MY_NOISE = 100 # 생성자가 가짜 데이터를 만들 때 입력으로 사용하는 노이즈 백터 숫자

MY_SHAPE = (28, 28, 1) # 손글씨 이미지의 모양, 흑백이고 화소수는 28*28, 채널은 1
MY_EPOCH = 5000
MY_BATCH = 300

# 출력 이미지 폴더 생성
MY_FOLDER = 'output/'
os.makedirs(MY_FOLDER,
            exist_ok=True)

for f in glob.glob(MY_FOLDER + '*'):
    os.remove(f)


############### 데이터 준비 ##############

# 결과는 numpy의 n-차원 행렬 형식
def red_data():
    # 학습용 입력값만 사용 (GAN은 비지도 학습이기 때문에 출력 데이터, 평가용도 필요 없음)
    (X_train, _), (_, _) = mnist.load_data()

    print('데이터 모양:', X_train.shape)
    # plt.imshow(X_train[0], cmap='gray')
    # plt.show()

    # [-1, 1] 데이터 스케일링
    X_train = X_train / 127.5 - 1.0

    # 채널 정보 추가
    X_train = np.expand_dims(X_train, axis=3)  # 차원 확장 함수 expand_dims
    print('데이터 모양:', X_train.shape)

    return X_train

# red_data()

################ 인공 신경망 구현 #################

# 생성자 설계
def build_generator():
    model = Sequential()

    # 입력층 + 은닉층 1
    model.add(Dense(MY_GEN,
                    input_dim=MY_NOISE))
    model.add(LeakyReLU(alpha=0.01))

    # 은닉층 2
    model.add(Dense(MY_GEN))
    model.add(LeakyReLU(alpha=0.01))

    # 은닉층 3 + 출력층
    # tanh 활성화는 [-1, 1] 스케일링 때문
    model.add(Dense(28 * 28 * 1,
                    activation='tanh'))
    model.add(Reshape(MY_SHAPE))

    print('\n생성자 요약')
    model.summary()

    return model

# build_generator()

# 감별자 설계
def build_discriminator():
    model = Sequential()

    # 입력층
    model.add(Flatten(input_shape=MY_SHAPE))

    # 은닉층 1
    model.add(Dense(MY_DIS))
    model.add(LeakyReLU(alpha=0.01))

    # 출력층
    model.add(Dense(1,
                    activation='sigmoid'))

    print('\n감별자 요약')
    model.summary()

    return model

# build_discriminator()


# DNN-GAN 구현
def build_GAN():
    model = Sequential()

    # 생성자 구현
    generator = build_generator()

    # 감별자 구현
    # 생성자 학습 시 감별자 고정
    discriminator = build_discriminator()

    discriminator.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['acc'])

    discriminator.trainable = False

    # GAN 구현: 생성자 먼저 추가, 그 다음 감별자
    model.add(generator)
    model.add(discriminator)

    # GAN은 정확도 무의미
    model.compile(optimizer='adam',
                  loss='binary_crossentropy')

    print('\nGAN 요약')
    model.summary()

    return discriminator, generator, model

build_GAN()









