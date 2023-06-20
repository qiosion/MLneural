from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 텐서플로우에서 신경망 모델 만들기
#  1. 모델을 만들고 Layer를 쌓는다
model = Sequential() # 모델을 만들어놓고 거기에 하나하나씩 레이어를 다는 것

# model.add(Dense(10, input_dim=2, activation="sigmoid"))
model.add(Dense(10, input_dim=2, activation="relu")) # 렐루함수 사용
model.add(Dense(2, activation="softmax")) # 여기서 input_dim은 안써줘도 자동으로 5
# Dense(출력의 갯수, input_dim=입력의 갯수, activation="활성화 함수")

"""
input_dim = input dimention
- 모델의 첫번째 레이어에는 반드시 input 의 크기가 들어가야함
- 그래서 Dense() 의 아규먼트로 몇개의 입력값을 집어넣을 것인지 알려줘야함

kernel_initializer 는 weight 의 초기값을 어떻게 줄 것인가를 설정함
"""

model.summary()

# 2. 모델의 최적화 옵션, 손실함수 등을 주고 컴파일한다
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# XOR 학습 데이터를 만들어보자
# 두 값이 같으면 클래스0이 1, 두 값이 다르면 클래스1이 1이 됨
x = [[0, 0], [0, 1], [1, 0], [1, 1]] # 입력
y = [[1, 0], [0, 1], [0, 1], [1, 0]] # 출력

# 학습
model.fit(x, y, epochs=1000)
# 데이터를 한번 쫙 넣어서 돌려본다 = 1번 돌린게 1 에포크 epoch(에폭)

# 모델이 예측한 결과를 보자
print(model.predict(x))







