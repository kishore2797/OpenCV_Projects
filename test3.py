import cv2 as cv
import numpy as np

img = cv.imread("test.jpg")

img0 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img1 = cv.Sobel(img0, cv.CV_64FC1, 0, 1, 7)

img2 = cv.Canny(img0, 100, 200)

cv.imshow("test", img)
cv.imshow("img0", img0)
cv.imshow("img2", img2)
cv.waitKey(0)
cv.destroyAllWindows()