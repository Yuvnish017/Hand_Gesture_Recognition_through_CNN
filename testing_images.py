import cv2 as cv
from keras.models import load_model
import numpy
import os

model = load_model('data_augmentation_model_hgr_v3.h5')
img_size = 64
background = numpy.ones((320, 320, 3), numpy.uint8)
background[:] = [255, 255, 255]
l_b = numpy.array([0, 32, 28])
u_b = numpy.array([72, 220, 170])

datadir = 'testing_images'

for img in os.listdir(datadir):
    image = cv.imread(os.path.join(datadir, img))
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
    print(roi.shape)
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    print(roi.shape)
    roi = cv.resize(roi, (125, 125))
    print(roi.shape)
    m = numpy.array(roi)
    print(m.shape)
    m = numpy.expand_dims(m, axis=0)
    print(m.shape)
    n = model.predict(m)
    print(n)
    max_value = max(n[0])
    z = [1 if i >= max_value else 0 for i in n[0]]
    i = z.index(1)
    if i == 0:
        print('A')
        cv.imwrite('A.jpg', image)
    if i == 1:
        print('B')
        cv.imwrite('B.jpg', image)
    if i == 2:
        print('C')
        cv.imwrite('C.jpg', image)
    if i == 3:
        print('D')
        cv.imwrite('D.jpg', image)
    if i == 4:
        print('E')
        cv.imwrite('E.jpg', image)
    if i == 5:
        print('F')
        cv.imwrite('F.jpg', image)
    if i == 6:
        print('G')
        cv.imwrite('G.jpg', image)
    if i == 7:
        print('H')
        cv.imwrite('H.jpg', image)
    if i == 8:
        print('I')
        cv.imwrite('I.jpg', image)
    if i == 9:
        print('J')
        cv.imwrite('J.jpg', image)
    if i == 10:
        print('K')
        cv.imwrite('K.jpg', image)
    if i == 11:
        print('L')
        cv.imwrite('L.jpg', image)
    if i == 12:
        print('M')
        cv.imwrite('M.jpg', image)
    if i == 13:
        print('N')
        cv.imwrite('N.jpg', image)
    if i == 14:
        print('O')
        cv.imwrite('O.jpg', image)
    if i == 15:
        print('P')
        cv.imwrite('P.jpg', image)
    if i == 16:
        print('Q')
        cv.imwrite('Q.jpg', image)
    if i == 17:
        print('R')
        cv.imwrite('R.jpg', image)
    if i == 18:
        print('S')
        cv.imwrite('S.jpg', image)
    if i == 19:
        print('T')
        cv.imwrite('T.jpg', image)
    if i == 20:
        print('U')
        cv.imwrite('U.jpg', image)
    if i == 21:
        print('V')
        cv.imwrite('V.jpg', image)
    if i == 22:
        print('W')
        cv.imwrite('W.jpg', image)
    if i == 23:
        print('X')
        cv.imwrite('X.jpg', image)
    if i == 24:
        print('Y')
        cv.imwrite('Y.jpg', image)
    if i == 25:
        print('Z')
        cv.imwrite('Z.jpg', image)
