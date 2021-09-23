import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_mask = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 7)


def nothing(x):
    pass


cv.namedWindow("Tracking")
cv.createTrackbar("size", "Tracking", 1, 20, nothing)
cv.createTrackbar("dynRatio", "Tracking", 1, 100, nothing)
cv.createTrackbar("thresh_1", "Tracking", 0, 30, nothing)
cv.createTrackbar("thresh_2", "Tracking", 0, 255, nothing)

while True:
    size = cv.getTrackbarPos("size", "Tracking")
    dynRatio = cv.getTrackbarPos("dynRatio", "Tracking")
    thresh_1 = cv.getTrackbarPos("thresh_1", "Tracking")
    thresh_2 = cv.getTrackbarPos("thresh_2", "Tracking")

    img_oil_paint = cv.xphoto.oilPainting(img, size, dynRatio, cv.COLOR_BGR2Lab)

    if thresh_1%2 == 0:
        thresh_1 += 1

    laplacian = cv.Laplacian(img, cv.CV_8UC1, ksize=thresh_1)

    laplacian = cv.cvtColor(laplacian, cv.COLOR_BGR2GRAY)

    cv.threshold(laplacian, thresh_2, 255, cv.THRESH_BINARY, laplacian)

    img4 = cv.bitwise_or(cv.cvtColor(cv.bitwise_not(laplacian), cv.COLOR_GRAY2BGR), img_oil_paint)

    cv.imshow("img4", img4)

    key = cv.waitKey(1)

    if key == 27:
        break

cv.destroyAllWindows()
