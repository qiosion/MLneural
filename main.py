import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    # gpu 쓰고있는지 확인
    print(tf.config.list_physical_devices("GPU"))
    print(device_lib.list_local_devices())