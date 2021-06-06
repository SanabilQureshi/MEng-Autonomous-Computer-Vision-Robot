import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import glob
import shutil
import image_edit
import time

fdirs = ['F:/### Sanabil Dissertation/Image Dataset/Blobs/Learning/Train',
         'F:/### Sanabil Dissertation/Image Dataset/Crack/Learning/Train',
         'F:/### Sanabil Dissertation/Image Dataset/Burns/Learning/Train']

NumpyData = 'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/NumpyData'

LABELS = ["Defect", "Non-Defect"]

input_files = glob.glob(str(NumpyData + '/*'))

for input_file in input_files:
    os.remove(input_file)

for fdir in fdirs:
    IMG_SIZE = 100
    Node_Inputs = []
    X = []
    y = []

    for category in LABELS:
        folder = os.path.join(fdir, category)
        all_images = glob.glob(str(f'{folder}' + "/*.png"))

        BinaryClass = LABELS.index(category)

        for img in tqdm(all_images):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            Node_Inputs.append([resized_image, BinaryClass])

    random.shuffle(Node_Inputs)

    for features, label in Node_Inputs:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    X_filename = f'X-Data-{fdir[-20:-15]}.npy'
    y_filename = f'y-Data-{fdir[-20:-15]}.npy'

    np.save(X_filename, X)
    np.save(y_filename, y)

    shutil.move(X_filename, NumpyData)
    shutil.move(y_filename, NumpyData)
