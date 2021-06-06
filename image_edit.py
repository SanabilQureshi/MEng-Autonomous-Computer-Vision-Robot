from PIL import Image
import numpy as np
import cv2 as cv2
import glob
import os
import shutil
import random as rd


def Any2PNG(input_loc, output_loc):
    To_PNG = glob.glob(str(f'{input_loc}' + "/*.jpg"))
    i = 1

    for img in To_PNG:
        png = cv2.imread(img)
        cv2.imwrite(str(f'{output_loc}' + f'./Converted - {os.path.basename(img)}'), png)
        i += 1


def image_input(input_loc):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(input_loc, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def Splitter(input_loc, output_loc, percentage):
    To_split = os.listdir(input_loc)
    Sorted = To_split[int(len(To_split) * 0): int(len(To_split) * percentage)]
    for img in Sorted:
        filepath = f'{input_loc}' + f'/{img}'
        shutil.copy(filepath, output_loc)


def Remover(input_loc):
    To_Remove = glob.glob(str(f'{input_loc}' + "/*.png"))
    for files in To_Remove:
        os.remove(files)


def Renamer(input_loc):
    To_Rename = glob.glob(str(f'{input_loc}' + "/*.png"))
    for img in To_Rename:
        i = rd.randrange(10000000, 100000000)
        j = rd.randrange(100000, 1000000)
        os.rename(img, f'{input_loc}/Cropped - {i}_{j} - Biscuit 6.png')


def Cropper(input_loc, output_loc):
    To_crop = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_crop:
        im = Image.open(img)
        boxes = ((300, 300, 2250, 2225),
                 (2550, 1175, 5955, 5120),
                 (145, 2195, 980, 3175),
                 (1025, 140, 1815, 1090),
                 (1015, 1185, 1855, 2125),
                 (1015, 2190, 1865, 3175))

        for tuple in boxes:
            region = im.crop(tuple)
            region.save(str(f'{output_loc}' + f'/Cropped - {os.path.basename(img)[12:-4]} - Biscuit {i}.png'))
            i += 1


def Padder(input_loc, output_loc, padding):
    To_pad = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_pad:
        pad = cv2.imread(img)
        height, width, centrepoint = pad.shape

        new_width = padding
        new_height = padding
        base_colour = (255, 255, 255)
        result = np.full((new_height, new_width, centrepoint), base_colour, dtype=np.uint8)

        new_x_dim = (new_width - width) // 2
        new_y_dim = (new_height - height) // 2

        result[new_y_dim:new_y_dim + height, new_x_dim:new_x_dim + width] = pad
        cv2.imwrite(str(f'{output_loc}' + f'./Padded - {os.path.basename(img)[10:]}'), result)
        i += 1


def Transparenter(input_loc, output_loc, unit_test=False):
    for dir in input_loc:
        To_transparent = glob.glob(str(f'{dir}' + "/*.png"))
        i = 1

        for img in To_transparent:
            pad = cv2.imread(img)
            transparented = cv2.cvtColor(pad, cv2.COLOR_BGR2BGRA)
            alpha = transparented[:, :, 3]
            alpha[np.all(transparented[:, :, 0:3] == (0, 0, 0), 2)] = 0
            cv2.imwrite(str(f'{output_loc}' + f'/Transparent - {os.path.basename(img)}'), transparented)


def Rescaler(input_loc, output_loc, scale_factor=1):
    for dir in input_loc:
        To_rescale = glob.glob(str(f'{dir}' + "/*.png"))
        i = 1

        for img in To_rescale:
            rescaled = cv2.imread(img)
            width = int(rescaled.shape[1] * scale_factor)
            height = int(rescaled.shape[0] * scale_factor)
            new_dimensions = width, height
            new_image = cv2.resize(rescaled, new_dimensions, interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(f'{output_loc}' + f'/Rescaled - {os.path.basename(img)}'), new_image)

            i += 1


def Circler(input_loc, output_loc, transparent=False):
    To_circle = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_circle:

        picture_rgb = cv2.imread(img, 1)
        picture = cv2.cvtColor(picture_rgb, cv2.COLOR_BGR2GRAY)

        blurred_picture = cv2.blur(picture, (20, 20))
        threshold, thresh = cv2.threshold(blurred_picture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh_inv = cv2.bitwise_not(thresh)
        stacked = np.dstack((thresh_inv, thresh_inv, thresh_inv))
        foreground = cv2.bitwise_and(picture_rgb, stacked)
        if transparent == True:
            transparented = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
            alpha = transparented[:, :, 3]
            alpha[np.all(transparented[:, :, 0:3] == (0, 0, 0), 2)] = 0
            cv2.imwrite(str(f'{output_loc}' + f'/Circled (T) -{os.path.basename(img)[9:]}'), transparented)
        else:
            cv2.imwrite(str(f'{output_loc}' + f'/Circled (T) -{os.path.basename(img)[9:]}'), foreground)
        i += 1


def Grabcutter(input_loc, output_loc, iterations=30, transparent=False):
    To_circle = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_circle:

        Biscuit_Grey = cv2.imread(img, 0)
        Biscuit_Colour = cv2.imread(img, 1)
        Image_Background = cv2.adaptiveThreshold(Biscuit_Grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 255)
        Biscuit_Grey = cv2.bitwise_not(Biscuit_Grey)

        Biscuit_Colour_Bilateral = cv2.bilateralFilter(Biscuit_Grey, 15, 5, 5)
        Biscuit_Colour_Binary, Biscuit_Colour_Otsu = cv2.threshold(Biscuit_Colour_Bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        Contour_Retrieve, Contour_ChainApprox = cv2.findContours(Biscuit_Colour_Otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Background_Copy = Image_Background.copy()
        Biscuit_Colour_Contour = cv2.drawContours(Background_Copy, Contour_Retrieve, -1, (0, 0, 0), 1)

        ## Grabcut implementation

        x, y, w, h = cv2.boundingRect(Contour_Retrieve[0])
        mask = np.zeros(Biscuit_Colour.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (x, y, w, h)

        cv2.grabCut(Biscuit_Colour, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        Result = Biscuit_Colour * mask2[:, :, np.newaxis]

        if transparent == True:
            transparented = cv2.cvtColor(Result, cv2.COLOR_BGR2BGRA)
            alpha = transparented[:, :, 3]
            alpha[np.all(transparented[:, :, 0:3] == (0, 0, 0), 2)] = 0
            cv2.imwrite(str(f'{output_loc}' + f'/Circled (G) - {os.path.basename(img)[10:]}'), transparented)
        else:
            cv2.imwrite(str(f'{output_loc}' + f'/Circled (G) - {os.path.basename(img)[10:]}'), Result)
        i += 1


def Rotator(input_loc, output_loc, step_size):
    To_rotate = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_rotate:
        rotate = cv2.imread(img)
        (h, w) = rotate.shape[:2]
        dimensions = (w, h)
        center = (w / 2, h / 2)

        for j in range(0, 360, step_size):
            M = cv2.getRotationMatrix2D(center, j, 1)
            rotated = cv2.warpAffine(rotate, M, dimensions)
            cv2.imwrite(str(f'{output_loc}' + f'/Rotated ({j}) degrees{os.path.basename(img)[7:]}'), rotated)
        i += 1


def Flipper(input_loc, output_loc, direction):
    To_flip = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_flip:
        aug = cv2.imread(img)
        for direction in ('vertical', 'horizontal', 'both'):
            if direction == 'vertical':
                label = 'vertical'
                flip = cv2.flip(aug, 0)
            elif direction == 'horizontal':
                label = 'horizontal'
                flip = cv2.flip(aug, 1)
            else:
                label = 'both'
                flip = cv2.flip(aug, -1)

            cv2.imwrite(str(f'{output_loc}' + f'/Flipped {os.path.basename(img)[8:]}'), flip)

        i += 1


def Noisy(input_loc, output_loc):
    To_noise = glob.glob(str(f'{input_loc}' + "/*.png"))
    i = 1

    for img in To_noise:
        pre_noise = cv2.imread(img)
        p = 255
        q = 100
        r = 50
        m = (p, q, r)
        s = (p, q, r)
        GNoise = cv2.randn(pre_noise, m, s)

        cv2.imwrite(str(f'{output_loc}' + f'/GNoise {os.path.basename(img)[8:]}'), GNoise)

        i += 1
