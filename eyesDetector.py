import sys
import numpy as np
import cv2

def convex_diff_test(x1, x2, d):
    """Test convex diff between values"""
    return ((x1+x2) > 0) and (((2*abs(x1 - x2))/(x1+x2)) < d)
#convex_diff_test

def absolute_area_test(area_cntr, area_img, d = 0.001):
    """Test size area to total image"""
    return ((area_img > 0) and ((area_cntr / area_img) > d))
#absolute_area_test

def absolute_position_test(circle, width, height, d = 0.38):
    """Test pupil placed at center of image"""
    (cx,cy),_ = circle
    return (np.sqrt((cx - .5*width)**2 + (cy - .5*height)**2) < (d*.5*np.sqrt(width**2 + height**2)))
#absolute_position_test

def circle_similarity_test(convexHullContour, d = 0.618):
    """Test circle similarity"""
    rect = cv2.minAreaRect(convexHullContour)
    a,b = rect[1]
    return ((min(a,b)/max(a,b)) > d)
#circle_similarity_test

def circle_area_test(convexHullContour, d = 0.2):
    """Test circle area difference"""
    _,r1 = cv2.minEnclosingCircle(convexHullContour)
    r2 = np.sqrt(cv2.contourArea(convexHullContour)/np.pi)
    return convex_diff_test(r1, r2, d)
#circle_area_test

def get_movement_circle(convexHullContour):
    """Return circle with center in center of mass and area equal to contour"""
    M = cv2.moments(convexHullContour)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    r = np.sqrt(cv2.contourArea(convexHullContour)/np.pi)
    return ((cx, cy), r)
#get_movement_circle

def int_circle(circle):
    (x,y),r = circle
    x=int(x)
    y=int(y)
    r=int(r)
    return ((x, y), r)
#int_circle

def absolute_border_test(circle, width, height):
    """Test pupil placed at center of image"""
    (cx,cy),r = circle
    ts = cy - r
    ls = cx - r
    bs = cy + r
    rs = cx + r
    return ((ts > 0) and (ls > 0) and (bs < height) and (rs < width))
#absolute_border_test

def pupil_detect(grayscale):
    height, width = grayscale.shape[:2]
    img_area = width * height
    res = None
    for i in range(0, 255):
        ret, thImage = cv2.threshold(grayscale, i, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key = cv2.contourArea)
        contours.reverse()
        for i in range(0, len(contours)):
            convexHull = cv2.convexHull(contours[i])
            a1 = cv2.contourArea(convexHull)
            a2 = cv2.contourArea(contours[i])
            if convex_diff_test(a1, a2, 0.2) and absolute_area_test(a2, img_area) and circle_similarity_test(convexHull) and circle_area_test(convexHull):
                circle = get_movement_circle(convexHull)
                if absolute_position_test(circle, width, height) and absolute_border_test(circle, width, height):
                    res = int_circle(circle),(cv2.contourArea(convexHull)/(cv2.arcLength(convexHull,True)**2))
                #^ if absolute_position_test(circle, width, height):
            #^ if convex_diff_test(a1, ...
        #^ for i in range(0, len(contours)):
    #^ for i in range(0, 255):
    return res
#pupil_detect

def get_img_pupil(img):
    img_inv_gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img_satr,_ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    res1 = pupil_detect(img_inv_gray)
    if res1 is not None:
        res2 = pupil_detect(img_satr)
        if res2 is not None:
            if res2[0][1] > res1[0][1]:
                t = res2
                res2 = res1
                res1 = t
            #
            if np.sqrt((res1[0][0][1] - res2[0][0][1])**2 + (res1[0][0][0] - res2[0][0][0])**2) < res2[0][1]:
                #res2 placed in res1
                return res2[0]
            elif res1[1] > res2[1]:
                return res1[0]
            else:
                return res2[0]
            #
        else:
            return res1[0]
        #
    else:
        res2 = pupil_detect(img_satr)
        if res2 is not None:
            return res2[0]
        else:
            return None
        #
    #
#get_img_pupil

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
            self.eyes = eye_cascade.detectMultiScale(self.roi_gray, 1.3, 5)
    
    def detectPupils(self):
        print("Eyes found: " + str(len(self.eyes)))
        if len(self.eyes) == 0:
            print("Unable to find eyes")
            quit()
        for (x,y,w,h) in self.eyes:
            img_eye = self.roi_color[y:y+h, x:x+w]
            
            circle = get_img_pupil(img_eye)
            if circle is not None:
                self.radiuses.append(circle[1])
                cv2.circle(img_eye, circle[0], circle[1], (255,0,0), 1)
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

    darkRadius = .5*(pupilsDetectorDark.radiuses[0] + pupilsDetectorDark.radiuses[1])
    lightRadius = .5*(pupilsDetectorLight.radiuses[0] + pupilsDetectorLight.radiuses[1])

    print("Dark pupil average radius: " + str(darkRadius))
    print("Light pupil average radius: " + str(lightRadius))

    ratio = darkRadius / lightRadius
    print("Ratio: " + str(ratio))
    if ratio >= 1.05:
        print("These eyes looks good")
    else:
        print("It looks like these eyes's owner a drug addictor")