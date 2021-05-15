import cv2 as cv
import numpy
from keras.models import load_model
from collections import deque

model = load_model('data_augmentation_model_hgr_v3.h5')
img_size = 125
background = numpy.ones((320, 320, 3), numpy.uint8)
background[:] = [255, 255, 255]
l_b = numpy.array([0, 32, 28])
u_b = numpy.array([72, 220, 170])
Q = deque(maxlen=64)
letters = []

cap = cv.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv.VideoWriter('output3.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    required_portion = frame[0:320, 320:640, :]
    cv.imshow('roi', required_portion)
    hsv = cv.cvtColor(required_portion, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, l_b, u_b)
    result = cv.bitwise_and(required_portion, required_portion, mask=mask)
    result2 = cv.bitwise_and(background, background, mask=mask)
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    j = result2[y:y + h, x:x + w, :]
    j = cv.resize(j, (img_size, img_size))
    j = cv.cvtColor(j, cv.COLOR_BGR2GRAY)
    cv.rectangle(frame, (320, 0), (640, 320), (0, 255, 255), 2)
    m = numpy.array(j)
    m = numpy.expand_dims(m, axis=0)
    n = model.predict(m)*100
    Q.append(n)
    res = numpy.array(Q).mean(axis=0)
    max_value = max(res[0])
    if max_value <= 20:
        cv.putText(frame, ' ', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    else:
        i = numpy.argmax(res)
        if i == 0:
            cv.putText(frame, 'A', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 1:
            cv.putText(frame, 'B', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 2:
            cv.putText(frame, 'C', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 3:
            cv.putText(frame, 'D', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 4:
            cv.putText(frame, 'E', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 5:
            cv.putText(frame, 'F', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 6:
            cv.putText(frame, 'G', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 7:
            cv.putText(frame, 'H', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 8:
            cv.putText(frame, 'I', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 9:
            cv.putText(frame, 'J', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 10:
            cv.putText(frame, 'K', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 11:
            cv.putText(frame, 'L', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 12:
            cv.putText(frame, 'M', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 13:
            cv.putText(frame, 'N', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 14:
            cv.putText(frame, 'O', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 15:
            cv.putText(frame, 'P', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 16:
            cv.putText(frame, 'Q', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 17:
            cv.putText(frame, 'R', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 18:
            cv.putText(frame, 'S', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 19:
            cv.putText(frame, 'T', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 20:
            cv.putText(frame, 'U', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 21:
            cv.putText(frame, 'V', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 22:
            cv.putText(frame, 'W', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 23:
            cv.putText(frame, 'X', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 24:
            cv.putText(frame, 'Y', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
        if i == 25:
            cv.putText(frame, 'Z', (150, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255))
    cv.imshow('frame', frame)
    out.write(frame)
    k = cv.waitKey(1)
    if k == 27:
        cv.destroyAllWindows()
        break

cap.release()
