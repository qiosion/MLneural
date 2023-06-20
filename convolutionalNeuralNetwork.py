import tensorflow as tf
from tensorflow.python.keras.models import Sequential
# convolution 2D
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense


# MNIST 데이터 가져오기
(train_data, train_label), (test_data, test_label) =\
tf.keras.datasets.mnist.load_data()

# 데이터 제대로 들어오는지 확인
print(train_data.shape) # (60000, 28, 28)
print(test_data.shape) # (10000, 28, 28)

# CNN 모델을 만들어보자
model = Sequential()

# 첫번째 conv layer
# 28x28 사이즈를 5x5에 맞추기 위해, 바깥영역을 0으로 채움 = zero padding
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                 activation="relu",
                 input_shape=(28, 28, 1),
                 padding="same"))
# 필터의 갯수 1개, 사이즈 5x5, 한칸씩(1x1) 이동, 활성화함수 relu
# input 값은 무조건 넣어줘야함. 근데 이미지라서 shape
# 2차원. 28x28 크기의 이미지. 흑백이라 세번째 인자에 1 넣어줌. 만약 값이 있다면 RGB 값이 있는거라 3차원임
# padding : 테두리에 패딩을 넣어줘서 크기가 줄어들지 않게함

# max pooling 2D -> 14x14 로 줄임
model.add(MaxPool2D(pool_size=(2, 2), # 4개의 영역에서 가장 큰 1개만 뽑아내고
                    strides=(2, 2), # 옆으로 두칸씩 이동
                    ))
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 1)         26        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 1)         0         
=================================================================
Total params: 26 # 학습해야하는 총 weight
Trainable params: 26
Non-trainable params: 0
_________________________________________________________________
"""

# 두번째 conv layer : 64개의 필터를 이용하여 5x5로 컨볼루션
model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
# 필터 1개, 크기(커널사이즈) 5x5 는 옵션명이 뭔지 생략 가능
# strides 없으면 기본(1, 1) 로 이동

"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 1)         26        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 1)         26        
=================================================================
Total params: 52 # 학습해야하는 총 weight
Trainable params: 52
Non-trainable params: 0
_________________________________________________________________
"""

# max pooling 2D -> 7x7로 줄임
model.add(MaxPool2D(pool_size=(2, 2), # 4개의 영역에서 가장 큰 1개만 뽑아내고
                    strides=(2, 2), # 옆으로 두칸씩 이동
                    ))
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 1)         26        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 1)         26        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 1)           0         
=================================================================
Total params: 52
Trainable params: 52
Non-trainable params: 0
_________________________________________________________________
"""

# flatten 한다 : 각 채널(64개)을 쭉 펴서 layer1로 만듦
model.add(Flatten())
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 1)         26        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 1)         26        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 1)           0         
_________________________________________________________________
flatten (Flatten)            (None, 49)                0         
=================================================================
Total params: 52
Trainable params: 52
Non-trainable params: 0
_________________________________________________________________
"""

# fully connected layer
model.add(Dense(1000, activation="relu"))
model.add(Dense(10, activation="softmax"))
# 마지막에 10개 나오게 하고, 그 합이 1이되게 함

"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 1)         26        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 1)         26        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 1)           0         
_________________________________________________________________
flatten (Flatten)            (None, 49)                0         
_________________________________________________________________
dense (Dense)                (None, 1000)              50000     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                10010     
=================================================================
Total params: 60,062
Trainable params: 60,062
Non-trainable params: 0
_________________________________________________________________
"""

# 모델 만들었으면 컴파일 하자
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

"""
# 학습 시키려 하면 오류남
model.fit(train_data, train_label, epochs=10, batch_size=200)
# 오류 원인
# 1. input dimension이 안맞음. 4차원이어야하는데 3차원을 넣었다고 오류생김. 채널정보를 따로 줘야함
# 2. train_label 이 숫자로 되어있는걸 처리 안함
"""

# 학습시키기 위해 모양 변경
train_data = train_data.reshape(60000, 28, 28, 1)
# train_data.shape[0] 을 통해 6만을 적어도 됨
# 1을 통해 흑백이라고 정확히 명시. 28x28x1

train_label = tf.keras.utils.to_categorical(train_label, 10)
# 정수형 레이블을 원-핫 인코딩 형식으로 변환
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 이런식으로 표시되게 함

# 학습
model.fit(train_data, train_label, epochs=10, batch_size=200)


