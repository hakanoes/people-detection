import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
people = cv2.CascadeClassifier("/home/hakanoes/people/haarcascade_fullbody.xml")
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))import imutils

while cap.isOpened():
    ret, frame = cap.read()
    # resize video
    resize = frame[200:800, 400:1500]
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fgmask = fgbg.apply(blur)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    body = people.detectMultiScale(fgmask, 1.3,  5)


    for (x, y, w, h) in body:
        cv2.rectangle(resize, (x, y), ((x + y), (w + h)), (0, 255, 0), 2)
        
    
    cv2.imshow("resize",fgmask)
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
