import os

import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# MNIST 손글씨 숫자 레이블
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 모델 로드
model = load_model('MNIST_CNN.hdf5')

# 이미지 로드
src = cv2.imread('img/number_image.png') # 20개중 4개틀림
# src = cv2.imread('img/9-0.png') # 이걸로하면 2개 틀림
# src = cv2.imread('img/0-9.png') # 2개 틀림
# print(src.shape) # (212, 823, 3) # BGR 순서
"""
src[100, 100] = [0, 0, 255] # 그림의 100,100 좌표에 빨간점찍혀있는거 확인가능

cv2.imshow("original image", src)
cv2.waitKey(0)
cv2.destroyAllWindows() # 아무 키나 누르면 창 꺼짐
# 아직 3차원 모델. 글자를 검은색으로 썼어도 RGB 값이 모두 있다
"""

# 그레이스케일 변환
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# print(gray.shape) # (212, 823)

gray = 255 - gray # 그레이 이미지 반전

"""
cv2.imshow("gray image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows() # 아무 키나 누르면 창 꺼짐
"""

# 이진화 Binarization

"""
# function name : my_threshold
# param in : grayImg
# param out : binImg
# Description
# 임계값 threshold 방법 : 밝기값이 일정 이상이면 1, 아니면 0
# 인자로 이미지나 threshold 를 받아서 이진화 하여 반환
def my_threshold(grayImg):
    binImg = grayImg.copy() # binary image
    threshold = 100 # 기준치 T 를 기준으로 이진

    for y in range(binImg.shape[0]):
        for x in range (binImg.shape[1]):
            if binImg[y, x] > threshold:
                binImg[y, x] = 255
            else:
                binImg[y, x] = 0
    return binImg
"""

# OpenCV 이진화 알고리즘
"""
- threshold(입력이미지, 임계값(T값), 최대값(임계값을 초과하는 픽셀의 값), 타입[, dst])
    - dst : 옵션. 결과 이미지를 어떤걸 사용할지
- threshold 사용하면 리턴값 2개나옴 !! 주의!
    - 1. 사용된 실제 임계값 return_value
    - 2. 이진화 결과 이미지
- rtn_val, binImg = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
- 임계값인 100을 기준으로 픽셀을 분류하여, 임계값보다 큰 픽셀은 흰색(255), 아니면 검은색(0)으로 할당
"""
rtn_val, binImg = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
# binImg : 이진화한 이미지
print('THRESH_OTSU : ', rtn_val)# 94.0

rtn_val, gray = cv2.threshold(gray, 94, 255, cv2.THRESH_TOZERO) # 교수님 사진
"""
# rtn_val, gray = cv2.threshold(gray, 115, 255, cv2.THRESH_TOZERO) # 내 사진
# rtn_val, gray = cv2.threshold(gray, 82, 255, cv2.THRESH_TOZERO) # 아이패드로 쓴 숫자
# 획은 살리고 배경을 0으로 만듦
# gray : mnist 데이터와 최대한 비슷하게 만든 이미지


# 연결된 요소 구하기
# n_blob, labelImg, stats, centroid = cv2.connectedComponentsWithStats(binImg)
n_blob, labelImg, stats, centroid = cv2.connectedComponentsWithStats(gray)
"""

# 팽창(Dilation) 연산을 적용하여 손글자를 두껍게 만듦
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(gray, kernel, iterations=1)

# 연결된 요소 구하기
n_blob, labelImg, stats, centroid = cv2.connectedComponentsWithStats(dilated)


gray_cp = gray.copy()

# 글자 크롭에 추가할 여유 공간
padding = 10

# 찾은 것을 그려보자. 지금은 원본 이미지를 사용한다
# cv2.rectangle(src, (100, 100, 200, 50), (255, 0, 255), thickness=2) # 사각형 그리기 예시
for i in range(1, n_blob): # 0번 블롭은 배경이기 때문에 안그린다
    x, y, w, h, area = stats[i] # 모든 stat에 대해 사각형 그리기
    cv2.rectangle(gray, (x, y, w, h), (255, 0, 255), thickness=2)
cv2.imshow("rectangle image", gray) # 전체 숫자에서 blob을 네모박스로 표시한 것
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 이미지 crop
# 이때 중심좌표가 [x, y]
숫자 하나만 크롭. 이때 stats[1] 는 6이다
x, y, w, h, area = stats[1] # 0은 배경덩어리라서 필요없음

# gray 에서 자르자. y부터 (y + h)까지, x부터 (x + w) 까지 자르자
crop = gray[y:y+h, x:x+w].copy() # 원본을 손상시키지 않으려면 카피해줘야한다.
cv2.imshow("crop image", crop) # stats[1] 크롭한 이미지 : 숫자 6

cv2.imshow("rectangle image", src) # 전체 숫자에서 blob을 네모박스로 표시한 것
cv2.waitKey(0)
cv2.destroyAllWindows() # 아무 키나 누르면 창 꺼짐
"""
# 블롭 좌표를 이용하여 모든 글자 크롭
for i in range(1, n_blob): # 0번 블롭은 배경이기 때문에 안그린다
    x, y, w, h, area = stats[i] # 블롭 좌표와 크기 추출

    # 좌우 여유 공간(padding) 추가하여 좌표 계산
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - 1, w + 2 * padding)
    h = min(gray.shape[0] - 1, h + 2 * padding)

    cv2.rectangle(gray_cp, (x, y, w, h), (255, 0, 255), thickness=2)
    # cv2.rectangle(gray_cp, (x,y), (x+w, y+h), (255, 0, 255), thickness=2)
    cv2.imshow("rectangle image", gray_cp)  # 전체 숫자에서 blob을 네모박스로 표시한 것
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    roi = gray_cp[y:y+h, x:x+w] # 이미지 크롭

    # 모델 사용하기
    """
    - 텐서플로우에 넣을 때 항상 데이터는 차원을 하나 늘려서 사용해야한다.
    - 따라서 크롭된 이미지를 [28x28]로 만든 다음, 차원을 하나 늘려줘서 모델에 넣어줘야 한다
    """

    # 1. 사이즈를 28x28로 바꾼다
    # resized_roi = cv2.resize(roi, (28, 28))
    resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA) # 비율 유지하면서 28x28로 변환

    # 2. 차원을 하나 늘린다
    img_data = np.expand_dims(resized_roi, axis=0)
    img_data = img_data.astype('float32') / 255.0

    pred = model.predict(img_data)
    pred_label = labels[np.argmax(pred)]
    print("예측값 : ", pred_label)
    # 이미지에 텍스트 표시
    # cv2.putText(gray_cp, str(pred_label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(src, str(pred_label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 결과 이미지 출력
for i in range(1, n_blob): # 0번 블롭은 배경이기 때문에 안그린다
    x, y, w, h, area = stats[i] # 모든 stat에 대해 사각형 그리기
    cv2.rectangle(src, (x, y, w, h), (255, 0, 255), thickness=2)
cv2.imshow('Image', src)
# cv2.imshow('Image', gray_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()

