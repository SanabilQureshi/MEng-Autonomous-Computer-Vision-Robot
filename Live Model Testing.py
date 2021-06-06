import cv2
import numpy as np
from urllib.request import urlopen
import time
import glob
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import GPU_RESET
import tensorflow as tf

GPU_RESET.reset_keras()

fdir = ['F:/Sanabil Dissertation/Image Dataset/Pre-processing/Cropped']

LABELS = ["Defect", "Non-Defect"]

model_blobs = tf.keras.models.load_model('./live models/Blobs')
model_cracks = tf.keras.models.load_model('./live models/Cracks')
model_burns = tf.keras.models.load_model('./live models/Burns')

url = 'http://192.168.137.129:8080/shot.jpg'


# url = 'http://192.168.1.191:8080/shot.jpg'

def nothing(x):
    pass


cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 640, 300)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 86, 86, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 54, 54, nothing)

cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    imgResp = urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    # result = cv2.bilateralFilter(result, 9, 75, 75)

    # cv2.imshow("frame", frame)
    # cv2.imshow("mask", mask)

    clean_result = result

    points = [[315, 490, 110, 275],
              [525, 690, 110, 275],
              [725, 890, 110, 275],
              [315, 490, 310, 475],
              [525, 690, 310, 475],
              [725, 890, 300, 465]]

    capture = cv2.waitKey(1)

    if capture == 99:
        IMG_SIZE = 100
        storage_array = []
        Blob_status = []
        Crack_status = []
        Burn_status = []
        for i in range(0, 6):
            test_raw = cv2.cvtColor(clean_result[points[i][2]:points[i][3], points[i][0]:points[i][1]], cv2.COLOR_BGR2GRAY)
            new_array = cv2.resize(test_raw, (IMG_SIZE, IMG_SIZE))
            test_reshape = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            # prediction = model.predict([test_reshape])
            Blob_predict = model_blobs.predict([test_reshape])
            Burn_predict = model_burns.predict([test_reshape])
            Crack_predict = model_cracks.predict([test_reshape])

            Blob_predict2 = Blob_predict[0][0]
            Crack_predict2 = Crack_predict[0][0]
            Burn_predict2 = Burn_predict[0][0]

            Blob_status.append(Blob_predict2)
            Burn_status.append(Burn_predict2)
            Crack_status.append(Crack_predict2)

        for item1 in Blob_status:
            item1 = np.round(item1, 4)
            item1 = item1 * 100
        for item2 in Burn_status:
            item2 = np.round(item2, 4)
            item2 = item2 * 100
        for item3 in Crack_status:
            item3 = np.round(item3, 4)
            item3 = item3 * 100

            output = np.max(storage_array)
            print(LABELS[int(output)])
            print(int(output))

    if capture == 115:
        IMG_SIZE = 100
        set = 0
        for i in range(0, 6):
            test_raw = cv2.cvtColor(clean_result[points[i][2]:points[i][3], points[i][0]:points[i][1]], cv2.COLOR_BGR2GRAY)
            new_array = cv2.resize(test_raw, (IMG_SIZE, IMG_SIZE))
            test_reshape = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            cv2.imwrite('Cropped (T) - ' + f'{set}' + '.png')

    for j in range(0, 6):
        cv2.rectangle(result, (points[j][0], points[j][2]), (points[j][1], points[j][3]), (73, 186, 0), 4)

    cv2.imshow("result", result)

    # time.sleep(0.01)

    if capture == 27:
        break

cv2.destroyAllWindows()