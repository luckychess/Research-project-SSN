import sys
import numpy as np
import cv2

class Detector(object):
    """Detecting eyes and pupils"""
    def __init__(self, img):
        self.img = img
        self.radiuses = []
        self.threshold_level_low = 230
        self.threshold_level_high = 231

    def detectEyes(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
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
        print("Eyes found: " + str(len(self.eyes)))
        if len(self.eyes) == 0:
            print("Unable to find eyes")
            quit()
        for (x,y,w,h) in self.eyes:
            img_eye = self.roi_gray[y:y+h, x:x+w]

            #self.showPicture(img_eye)
            
            for i in range(self.threshold_level_low, self.threshold_level_high):
                inv_eye = ~img_eye
                ret, thImage = cv2.threshold(inv_eye, i, 255, cv2.THRESH_BINARY)
    
                _, contours, hierarchy = cv2.findContours(thImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(inv_eye, contours, -1, (255,255,255), -1)

                for j in range (0, len(contours)):
                    area = cv2.contourArea(contours[j])
                    xr,yr,wr,hr = cv2.boundingRect(contours[j])
                    radius = wr/2

                    # If contour is big enough and has round shape
                    # Then it is the pupil
                    if area >= 60 and abs(1 - (wr / hr)) <= 0.2 and abs(1 - (area / (np.pi * radius**2))) <= 0.5:
                        self.radiuses.append(radius)
                        print("Thresold level: " + str(i) + ", Area: " + str(area) + ", width: " + str(wr) + ", height: " + str(hr) + ", radius: " + str(radius))
                        cv2.circle(img_eye, (int(xr + radius), int(yr + radius)), int(radius), (255,0,0), 1)
                        self.showPicture(img_eye)
   
    def showPicture(self, eye):
        cv2.imshow('Eyes', eye)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eyesDetector.py <dark_picture_name> <light_picture_name>")
        quit()
    
    img_dark = cv2.imread(sys.argv[1])
    if img_dark is None:
        print("Unable to open file " + sys.argv[1])
        quit()

    img_light = cv2.imread(sys.argv[2])
    if img_light is None:
        print("Unable to open " + sys.argv[2])
        quit()

    pupilsDetectorDark = Detector(img_dark)
    pupilsDetectorDark.detectEyes()
    pupilsDetectorDark.detectPupils()

    pupilsDetectorLight = Detector(img_light)
    pupilsDetectorLight.detectEyes()
    pupilsDetectorLight.detectPupils()

    darkToLightRadius1 = pupilsDetectorDark.radiuses[0] / pupilsDetectorLight.radiuses[0]
    darkToLightRadius2 = pupilsDetectorDark.radiuses[1] / pupilsDetectorLight.radiuses[1]

    print("First eye ratio: " + str(darkToLightRadius1))
    print("Second eye ratio: " + str(darkToLightRadius2))

    averageRatio = (darkToLightRadius1 + darkToLightRadius2) / 2
    print("Both eyes average: " + str(averageRatio))
    if averageRatio >= 1.05:
        print("These eyes looks good")
    else:
        print("It looks like these eyes's owner a drug addictor")