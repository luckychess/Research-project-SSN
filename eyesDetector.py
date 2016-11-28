import numpy as np
import cv2

class Detector(object):
    """Detecting eyes and pupils"""
    def __init__(self, img):
        self.img = img

    def detectEyes(self):
        face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
    
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(self.gray, 1.3, 5)
        if len(faces) == 0:
            print("Unable to find any faces on the picture")
            quit()
        
        for (x,y,w,h) in faces:
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            self.roi_gray = self.gray[y:y+h, x:x+w]
            self.roi_color = self.img[y:y+h, x:x+w]
            self.eyes = eye_cascade.detectMultiScale(self.roi_gray)

    def detectPupils(self):
        for (x,y,w,h) in self.eyes:
            img_eye = self.roi_gray[y:y+h, x:x+w]
            inv_eye = ~img_eye

            ret, thImage = cv2.threshold(inv_eye, 220, 255, cv2.THRESH_BINARY)

            _, contours, hierarchy = cv2.findContours(thImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            print("Contours found: " + str(len(contours)))
            cv2.drawContours(inv_eye, contours, -1, (255,255,255), -1)

            for i in range (0, len(contours)):
                area = cv2.contourArea(contours[i])
                xr,yr,wr,hr = cv2.boundingRect(contours[i])
                radius = wr/2

                # If contour is big enough and has round shape
                # Then it is the pupil
                if area >= 60 and abs(1 - (wr / hr)) <= 0.2 and abs(1 - (area / (np.pi * radius**2))) <= 0.5:
                    print("Area: " + str(area) + " width: " + str(wr) + " height: " + str(hr))
                    cv2.circle(img_eye, (int(xr + radius), int(yr + radius)), int(radius), (255,0,0), 1)
                    self.showPicture(img_eye)
   
    def showPicture(self, eye):
        cv2.imshow('Eyes', eye)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread('eyes_ideal.jpg')
    pupilsDetector = Detector(img)
    pupilsDetector.detectEyes()
    pupilsDetector.detectPupils()
