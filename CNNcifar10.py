import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt


# cifar10 데이터 가져오기
(train_data, train_label), (test_data, test_label) =\
tf.keras.datasets.cifar10.load_data()

# 데이터 제대로 들어오는지 확인
print(train_data.shape) # (50000, 32, 32, 3)
print(test_data.shape) # (10000, 32, 32, 3)

# label 이 숫자로 되어있어서 헷갈리다. 귀찮지만 글자로 적어주자
class_name = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# 모델 구현
# (conv + max-pool) * 3
model = Sequential()

# 1st layers 첫번째 반복
model.add(Conv2D(32, (5, 5), activation="relu", #
                 input_shape=(32, 32, 3), # 32x32x3채널(칼라). 처음이니까 input_shape 적어줘야함
                 padding="same")) # 줄어들지 않게 함
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        2432      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
=================================================================
Total params: 2,432
Trainable params: 2,432
Non-trainable params: 0
"""

# 2st layers 두번째 반복
model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        2432      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 32)        25632     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
=================================================================
Total params: 28,064
Trainable params: 28,064
Non-trainable params: 0
"""

# 3st layers 세번째 반복
model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        2432      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 32)        25632     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
=================================================================
Total params: 79,328
Trainable params: 79,328
Non-trainable params: 0
"""

# full connected layer 연결
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

"""
model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        2432      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 32)        25632     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          51264     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                65600     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 145,578
Trainable params: 145,578
Non-trainable params: 0
_________________________________________________________________
"""

# 컴파일
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 학습할 수 있도록 모양 변경
train_label = tf.keras.utils.to_categorical(train_label, 10)

# 학습
model.fit(train_data, train_label, epochs=10, batch_size=200)

# 테스트
test_label = tf.keras.utils.to_categorical(test_label, 10)
model.evaluate(test_data, test_label)


