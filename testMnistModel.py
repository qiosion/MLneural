import tensorflow as tf

clf = tf.keras.models.load_model("first_mnist.h5")

# 학습된 모델 로딩
(train_set, train_label), (test_set, test_label) = tf.keras.datasets.mnist.load_data()

# 학습할 때 1차원으로 했으니, 테스트 데이터의 모양을 reshape 해준다
test_set = test_set.reshape(10000, 784)
# 샘플이 하나여도 2차원 배열을 넣어야 함
# 결과는 1차원 배열로 나옴
# print(clf.predict(test_set[:1])) # 테스트 샘플 하나를 predict

# 모든 샘플에 대해 틀린 것 확인
for i in range(len(test_set)): # 1만개의 데이터를 for루프 돈다
    result = clf.predict(test_set[i:i+1]) # 2차원
    print(result)

    # 이미지 그리기
    print(test_label[i]) # test의 정답을 vector로 보여줌
    img = test_set[i].reshape(28, 28) # 2차원으로 reshape
    # 정답 인덱스를 argmax 이용해서 찾아보자









