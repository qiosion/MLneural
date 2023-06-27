import os

import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# MNIST 손글씨 숫자 레이블
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 모델 로드
model = load_model('MNIST_CNN.hdf5')

# 1. 이미지 로드 및 그레이스케일 변환
image  = cv2.imread('img/all_num.jpg', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화를 위한 임계값 설정 (128 하니까 잘안돼)
threshold = 170

ret , binary = cv2.threshold(gray_image,threshold,255,cv2.THRESH_BINARY_INV) #영상 이진화 + 반전

cv2.imshow('binary',binary)

k = cv2.waitKey(0)

cv2.destroyAllWindows()



############################

# Blob 검출을 위한 파라미터 설정
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.maxArea = 5000
params.filterByCircularity = False  # 원형 필터링 비활성화

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(binary)

# 문자 영역 처리 및 인식
for keypoint in keypoints:
    # 문자 영역 좌표 및 크기 추출
    x = int(keypoint.pt[0] - keypoint.size / 2)
    y = int(keypoint.pt[1] - keypoint.size / 2)
    width = int(keypoint.size)
    height = int(keypoint.size)

    # 문자 영역 추출
    roi = image[y:y+height, x:x+width]

    # 모델 입력 형식에 맞게 변환
    image_data = np.expand_dims(roi, axis=0)
    image_data = np.resize(image_data, (1, 28, 28, 1))
    image_data = image_data.astype('float32') / 255.0

    # 숫자 인식 및 결과 출력
    prediction = model.predict(image_data)
    predicted_label = labels[np.argmax(prediction)]
    print("Predicted label:", predicted_label)

# 결과 이미지에 문자 영역과 인식 결과 표시
image_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Image', image_with_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
# Blob 중심 좌표와 경계 사각형 추출
centers = []
rectangles = []
for keypoint in keypoints:
    center = keypoint.pt
    size = keypoint.size
    x = int(center[0] - size/2)
    y = int(center[1] - size/2)
    width = int(size)
    height = int(size)
    rect = (x, y, width, height)
    centers.append(center)
    rectangles.append(rect)

# 글자 크기에 맞는 필터링 수행
filtered_centers = []
filtered_rectangles = []
for center, rect in zip(centers, rectangles):
    x, y, width, height = rect
    # 글자 크기에 맞는 조건 설정 (예시)
    if width > 10 and height > 10:
        filtered_centers.append(center)
        filtered_rectangles.append(rect)

# Blob 중심 좌표를 기준으로 숫자 인식
for center, rect in zip(filtered_centers, filtered_rectangles):
    x, y, width, height = rect
    roi = binary[y:y+height, x:x+width]  # ROI 추출, MNIST 손글씨와 동일한 크기인 28x28로 설정
    roi = cv2.resize(roi, (28, 28))  # MNIST 모델에 입력할 크기로 변환
    roi = np.expand_dims(roi, axis=2)
    roi = np.expand_dims(roi, axis=0)
    roi = roi.astype('float32') / 255.0

    # 숫자 인식 및 결과 출력
    prediction = model.predict(roi)
    predicted_label = labels[np.argmax(prediction)]
    print("Predicted label:", predicted_label)

# 결과 이미지에 Blob 표시
image_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Image with Blobs', image_with_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""






"""


binary = cv2.morphologyEx(binary , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)

cv2.imshow('binary',binary)

k = cv2.waitKey(0)

cv2.destroyAllWindows()

########

contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
#외곽선 검출
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) #이진화 이미지를 color이미지로 복사(확인용)
cv2.drawContours(color , contours , -1 , (0,255,0),3) #초록색으로 외곽선을 그려준다.

#리스트연산을 위해 초기변수 선언
bR_arr = []
digit_arr = []
digit_arr2 = []
count = 0

#검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)) :
    bin_tmp = binary.copy()
    x,y,w,h = cv2.boundingRect(contours[i])
    bR_arr.append([x,y,w,h])

print(bR_arr[:5])

#x값을 기준으로 배열을 정렬
bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False)

print(bR_arr[:5])

print(len(bR_arr))

# 작은 노이즈데이터 버림,사각형그리기,12개씩 리스트로 다시 묶어서 저장
for x, y, w, h in bR_arr:
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]
    if tmp_x and tmp_y > 10:
        count += 1
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])
        if count == 12:
            digit_arr2.append(digit_arr)
            digit_arr = []
            count = 0

cv2.imshow('contours', color)

k = cv2.waitKey(0)
cv2.destroyAllWindows()


#### 여기부터 밑에 안됨
#리스트에 저장된 이미지를 28x28의 크기로 리사이즈해서 순서대로 저장
for i in range(0,len(digit_arr2)) :
    for j in range(len(digit_arr2[i])) :
        count += 1
        if i == 0 :         #1일 경우 비율 유지를 위해 마스크를 만들어 그위에 얹어줌
            width = digit_arr2[i][j].shape[1]
            height = digit_arr2[i][j].shape[0]
            tmp = (height - width)/2
            mask = np.zeros((height,height))
            mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
            digit_arr2[i][j] = cv2.resize(mask,(28,28))
        else:
            digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(28,28))
        if i == 9 : i = -1
        cv2.imwrite('img/'+str(i+1)+'_'+str(j)+'.jpg',digit_arr2[i][j])

####### 위 코드 안됨

def createdataset(directory):  # sklearn사용을 위해 데이터세트를 생성
    files = os.listdir(directory)
    x = []
    y = []
    for file in files:
        attr_x = cv2.imread(directory + file, cv2.IMREAD_GRAYSCALE)
        ret, attr_x = cv2.threshold(attr_x, 170, 255, cv2.THRESH_BINARY)  # 영상 이진화
        attr_x = attr_x.reshape(28, 28, 1)
        attr_y = int(file[0])

        x.append(attr_x)
        y.append(attr_y)

    x = np.array(x)

    y = np.array([[i] for i in y])
    enc = OneHotEncoder(categories='auto')
    enc.fit(y)
    y = enc.transform(y).toarray()

    return x, y

test_dir = 'img/'
test_x, test_y = createdataset(test_dir)

print('data set shape :', test_x.shape, test_y.shape)


"""


