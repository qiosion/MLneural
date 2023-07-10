# mnist 데이터 세트를 이용하여 cnn 모델을 학습하시오

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


# MNIST 데이터 가져오기
(train_data, train_label), (test_data, test_label) =\
tf.keras.datasets.mnist.load_data()

# 학습시키기 위해 모양 변경
train_data = train_data.reshape(60000, 28, 28, 1).astype('float32') / 255
train_label = tf.keras.utils.to_categorical(train_label, 10)
# 테스트를 위해 모양 변경
test_data = test_data.reshape(10000, 28, 28, 1).astype('float32') / 255
test_label = tf.keras.utils.to_categorical(test_label, 10)

# 데이터 제대로 들어오는지 확인
print(train_data.shape) # (60000, 28, 28)
print(test_data.shape) # (10000, 28, 28)

# CNN 모델을 만들어보자
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), # strides=(1, 1),
                 activation="relu",
                 input_shape=(28, 28, 1)
                 ))
                 #padding="same"))

model.add(Conv2D(filters=32, kernel_size=(3, 3), # strides=(1, 1),
                 activation="relu"
                 ))

model.add(Conv2D(filters=64, kernel_size=(3, 3), # strides=(1, 1),
                 activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten()) # 채널을 쭉 펴서 layer1로 만듦

# fully connected layer
model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax")) # 마지막에 10개 나오게 하고, 그 합이 1이되게 함

# 모델 컴파일
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 모델 최적화 구간
modelpath = './MNIST_CNN.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 학습
history = model.fit(train_data, train_label, epochs=30, batch_size=200,
                    validation_split=0.25,
                    verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

# 테스트
test_loss, test_accuracy = model.evaluate(test_data, test_label)

print("test_loss : ", test_loss) # 0.03104587458074093
print("test_accuracy : ", test_accuracy) # 0.9919999837875366