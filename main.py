import cv2
import numpy as np


class PerspectiveCorrection:
    def __init__(self, img):
        self.img = img
        self.tump = []


    def edge_detection(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 130, 250)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        self.img_contours = self.img.copy()
        cv2.drawContours(self.img_contours, contours[0], -1, (0, 255, 0), 3)
        return contours
        
    def array_sorting(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 
        arr = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)
        column = arr[:, 0]
        rect[1] = arr[np.argmax(column)]
        rect[3] = arr[np.argmin(column)]  
        return rect
    
    def perspective(self, rect):
        width, height = 400, 500
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(rect, pts2)
        self.warped = cv2.warpPerspective(self.img, matrix, (width, height))
    
    def corner_detection(self, contours):
        peri = cv2.arcLength(contours[0], True)  
        epsilon = 0.02 * peri                 
        approx = cv2.approxPolyDP(contours[0], epsilon, True)  
        for point in approx:
            x, y = point[0] 
            self.tump.append((x, y))
            
            cv2.circle(self.img_contours, (x, y), radius=5, color=(0, 0, 255), thickness=-1) 
        
    def main(self):
        contours = self.edge_detection()
        self.corner_detection(contours)
        
        if 4 == len(self.tump):
            pts = np.array(self.tump, dtype="float32")
            rect = self.array_sorting(pts)
            self.perspective(rect)
            cv2.imshow('Perspektif', self.warped)
            cv2.imwrite("output.jpg", self.warped)
            
        else:
            print("pespektif için 4 köşeye ihtiyaç vardır")
        
        cv2.imshow('Goruntu', self.img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('input.jpg')
    process = PerspectiveCorrection(img)
    process.main()


