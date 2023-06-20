import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib # 모델 저장
# from tensorflow.keras.models import Sequential # 순차모델
# from tensorflow.keras.layers import Dense # 덴스레이어
# from tensorflow.keras.models import load_model # 모델 가지고오기


# 리턴값이 4개. 받아온 그대로의 형태를 가져야 하므로 튜플() 로 짝지어줘야함
(train_data, train_label), (test_data, test_label) =\
tf.keras.datasets.mnist.load_data()

"""
print(train_data.shape) # (60000, 28, 28) # 이미지가 28px * 28px 라는 뜻. 흑백이미지.
# 칼라이미지라면 3채널이므로 (60000, 28, 28, 3) 라고 떴을 것
print(train_label.shape) # (60000,)
print(test_data.shape) # (10000, 28, 28)
print(test_data.shape) # (10000, 28, 28)

print(train_label[:10]) # [5 0 4 1 9 2 1 3 1 4]
"""

# print(train_data[0]) # 숫자 5
#
# plt.imshow(train_data[0]) # 이미지로 띄워보자
# plt.show()

# 함수로 만들어보자
def show_image(img):
    plt.imshow(img, cmap="gray") # 흑백으로 출력
    # plt.imshow(255-img, cmap="gray") # 보기 편하게 바꿈. 실제로는 이거안씀
    plt.show()

"""
# 이미지로 확인
index = 0
while True: # 종료하려면 우측상단 stop 버튼
    print("train_label[index]", train_label[index])
    show_image(train_data[index])
    index += 1
"""

"""
# 데이터 분포 확인
print(type(train_label)) # <class 'numpy.ndarray'>

count = np.bincount(train_label) # 주머니가 몇갠지 세어줌
print(count) # [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]

plt.bar(np.arange(0, 10), count)
plt.show()
"""

# 랜덤포레스트 학습 테스트
# clf = RandomForestClassifier() # 모델 저장 후 필요없어짐

clf = joblib.load("rf_mnist.pkl") # 모델 불러오기

# train_data = 6만 * 28 * 28 라서 3차원임
# train_data = 1만 * 28 * 28 => 3차원임

# 즉, fit (학습) 할때 그냥 train_data 와 test_data 를 넣으면 에러
# Found array with dim 3. RandomForestClassifier expected <= 2

# 2차원 배열을 1차원으로 reshape 한다
train_data = train_data.reshape(60000, 784) # 28 * 28
test_data = test_data.reshape(10000, 784)

# 모델 저장 후 필요없어짐
# clf.fit(train_data, train_label) # 학습하여 모델 생김

# 모델의 저장
# joblib.dump(clf, "rf_mnist.pkl") # clf을 rf_mnist.pkl라는 이름의 파일로 저장

# 모델이 저장된 이후라면 clf의 정의와 학습(fit)이 필요없어짐
# 대신 위에서 모델을 불러오자

# test_data에 대한 검증
# print(clf.predict(test_data[:10]))
# print(test_label[:10])

# print(clf.score(test_data, test_label))

# MLP로 학습 해보기
# one-hot encoding : 출력노드 10개 중 내가 원하는 target 만 활성화(1) 되게 함
# 정답은 원핫인코딩 형태로 출력한다
train_label = tf.keras.utils.to_categorical(train_label, 10)
test_label = tf.keras.utils.to_categorical(test_label, 10)


def make_model(): # 모델 설계를 함수로 만들자
    # 모델 설계
    model = tf.keras.models.Sequential()

    hidden1 = tf.keras.layer.Dense(512, input_dim=784, activation="relu")
    # 은닉층1은 덴스 레이어로 하겠다
    # 모델 입력 : 784 차원
    # 모델 출력 : 512 ..... 는 도대체 뭐.. 어디서나온걸까
    # w 갯수 = 파라미터 = 784 * 512 + bias
    # 활성화함수 : 모르겠으면 relu

    model.add(hidden1) # 모델에 넣을 때 layer를 상속한 애를 넣어준다

    outlayer = tf.keras.layer.Dense(10, activation="softmax")
    # 모델 출력 : 10 개
    # softmax : 결과값의 합이 1이 되게함. 정형화시킴
    # 출력이 발산하는 값으로 나오면, 스케일이 달라서 코드를 처리하기 힘듦
    # w 갯수 = 학습해야할 파라미터 = 10 * 512 + bias

    model.add(outlayer)

    model.summary()
    # 총 학습해야할 파라미터 = 407,050

    # 모델 컴파일
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    # 모델 학습
    model.fit(train_data, train_label, epochs=30, batch_size=200)
    model.save("first_mnist.h5") # 모델 저장

    return model
# make_model()

# 모델 가지고오기
mlp = tf.keras.models.load_model("first_mnist.h5")
# print(mlp.predict(test_data[:1]))

# 모델에 테스트 데이터를 통째로 던져주면?
# print(mlp.evaluate(test_data, test_label))
# 손실 loss 와 정답률 accuracy [0.7744832038879395, 0.9769999980926514]

# 과제 : 테스트 -> 모델 돌렸을 때 어디가 틀린 데이터로 뜨는지 인덱스 확인하기
print(mlp.predict(test_data[7:8])) # 2차원으로 넣어줘야함
# 7번 데이터를 넣었더니 인덱스 9가 뜸

ans = mlp.predict(test_data[7:8])

"""이하 불확실
argmax = max(ans)
print(argmax)
"""
