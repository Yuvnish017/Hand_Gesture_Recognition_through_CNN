import numpy
import cv2 as cv
import os
import csv


def create_file_list(my_dir, format='.jpg'):
    file_list = []
    print(my_dir)
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list


datadir = 'datasets'
img_size = 28
background = numpy.ones((320, 320, 3), numpy.uint8)
background[:] = [255, 255, 255]
l_b = numpy.array([0, 32, 28])
u_b = numpy.array([72, 220, 170])

for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']:
    path = os.path.join(datadir, i)
    for file in os.listdir(path):
        image = cv.imread(os.path.join(path, file))
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, l_b, u_b)
        res = cv.bitwise_and(image, image, mask=mask)
        res2 = cv.bitwise_and(background, background, mask=mask)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        roi = res2[y:y + h, x:x + w, :]
        roi = cv.resize(roi, (img_size, img_size))
        value = numpy.asarray(roi, dtype=numpy.int)
        value = value.flatten()
        if i == 'A':
            value = numpy.insert(value, 0, 0, axis=0)
        if i == 'B':
            value = numpy.insert(value, 0, 1, axis=0)
        if i == 'C':
            value = numpy.insert(value, 0, 2, axis=0)
        if i == 'D':
            value = numpy.insert(value, 0, 3, axis=0)
        if i == 'E':
            value = numpy.insert(value, 0, 4, axis=0)
        if i == 'F':
            value = numpy.insert(value, 0, 5, axis=0)
        if i == 'G':
            value = numpy.insert(value, 0, 6, axis=0)
        if i == 'H':
            value = numpy.insert(value, 0, 7, axis=0)
        if i == 'I':
            value = numpy.insert(value, 0, 8, axis=0)
        if i == 'J':
            value = numpy.insert(value, 0, 9, axis=0)
        if i == 'K':
            value = numpy.insert(value, 0, 10, axis=0)
        if i == 'L':
            value = numpy.insert(value, 0, 11, axis=0)
        if i == 'M':
            value = numpy.insert(value, 0, 12, axis=0)
        if i == 'N':
            value = numpy.insert(value, 0, 13, axis=0)
        if i == 'O':
            value = numpy.insert(value, 0, 14, axis=0)
        if i == 'P':
            value = numpy.insert(value, 0, 15, axis=0)
        if i == 'Q':
            value = numpy.insert(value, 0, 16, axis=0)
        if i == 'R':
            value = numpy.insert(value, 0, 17, axis=0)
        if i == 'S':
            value = numpy.insert(value, 0, 18, axis=0)
        if i == 'T':
            value = numpy.insert(value, 0, 19, axis=0)
        if i == 'U':
            value = numpy.insert(value, 0, 20, axis=0)
        if i == 'V':
            value = numpy.insert(value, 0, 21, axis=0)
        if i == 'W':
            value = numpy.insert(value, 0, 22, axis=0)
        if i == 'X':
            value = numpy.insert(value, 0, 23, axis=0)
        if i == 'Y':
            value = numpy.insert(value, 0, 24, axis=0)
        if i == 'Z':
            value = numpy.insert(value, 0, 25, axis=0)
        with open("train_dataset.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)

for i in ['A_test', 'B_test', 'C_test', 'D_test', 'E_test', 'F_test', 'G_test', 'H_test', 'I_test', 'J_test', 'K_test',
          'L_test', 'M_test', 'N_test', 'O_test', 'P_test', 'Q_test', 'R_test', 'S_test', 'T_test', 'U_test', 'V_test',
          'W_test', 'X_test', 'Y_test', 'Z_Test']:
    path = os.path.join(datadir, i)
    for file in os.listdir(path):
        image = cv.imread(os.path.join(path, file))
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, l_b, u_b)
        res = cv.bitwise_and(image, image, mask=mask)
        res2 = cv.bitwise_and(background, background, mask=mask)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, 0)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        roi = res2[y:y + h, x:x + w, :]
        roi = cv.resize(roi, (img_size, img_size))
        value = numpy.asarray(roi, dtype=numpy.int)
        value = value.flatten()
        if i == 'A_test':
            value = numpy.insert(value, 0, 0, axis=0)
        if i == 'B_test':
            value = numpy.insert(value, 0, 1, axis=0)
        if i == 'C_test':
            value = numpy.insert(value, 0, 2, axis=0)
        if i == 'D_test':
            value = numpy.insert(value, 0, 3, axis=0)
        if i == 'E_test':
            value = numpy.insert(value, 0, 4, axis=0)
        if i == 'F_test':
            value = numpy.insert(value, 0, 5, axis=0)
        if i == 'G_test':
            value = numpy.insert(value, 0, 6, axis=0)
        if i == 'H_test':
            value = numpy.insert(value, 0, 7, axis=0)
        if i == 'I_test':
            value = numpy.insert(value, 0, 8, axis=0)
        if i == 'J_test':
            value = numpy.insert(value, 0, 9, axis=0)
        if i == 'K_test':
            value = numpy.insert(value, 0, 10, axis=0)
        if i == 'L_test':
            value = numpy.insert(value, 0, 11, axis=0)
        if i == 'M_test':
            value = numpy.insert(value, 0, 12, axis=0)
        if i == 'N_test':
            value = numpy.insert(value, 0, 13, axis=0)
        if i == 'O_test':
            value = numpy.insert(value, 0, 14, axis=0)
        if i == 'P_test':
            value = numpy.insert(value, 0, 15, axis=0)
        if i == 'Q_test':
            value = numpy.insert(value, 0, 16, axis=0)
        if i == 'R_test':
            value = numpy.insert(value, 0, 17, axis=0)
        if i == 'S_test':
            value = numpy.insert(value, 0, 18, axis=0)
        if i == 'T_test':
            value = numpy.insert(value, 0, 19, axis=0)
        if i == 'U_test':
            value = numpy.insert(value, 0, 20, axis=0)
        if i == 'V_test':
            value = numpy.insert(value, 0, 21, axis=0)
        if i == 'W_test':
            value = numpy.insert(value, 0, 22, axis=0)
        if i == 'X_test':
            value = numpy.insert(value, 0, 23, axis=0)
        if i == 'Y_test':
            value = numpy.insert(value, 0, 24, axis=0)
        if i == 'Z_test':
            value = numpy.insert(value, 0, 25, axis=0)
        with open("test_dataset.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
