import numpy as np
import cv2

class Detector(object):
    """Detecting eyes and pupils"""
    def __init__(self, img):
        self.img = img

    def detectEyes(self):
        face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
    
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            self.roi_gray = gray[y:y+h, x:x+w]
            self.roi_color = img[y:y+h, x:x+w]
            self.eyes = eye_cascade.detectMultiScale(self.roi_gray)
   
    def showPicture(self):
        for (ex,ey,ew,eh) in self.eyes:
            cv2.rectangle(self.roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('Eyes', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    pupilsDetector = Detector(img)
    pupilsDetector.detectEyes()
    pupilsDetector.showPicture()
