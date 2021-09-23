import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")
img1 = cv.imread("2_2.jpg")

img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img3 = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 8)

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.waitKey(0)
# cv.destroyAllWindows()


def nothing(x):
    pass


cv.namedWindow("Tracking")
cv.createTrackbar("size", "Tracking", 0, 3, nothing)
cv.createTrackbar("orig_ksize", "Tracking", 0, 30, nothing)
cv.createTrackbar("ksize", "Tracking", 0, 60, nothing)

while True:
    size = cv.getTrackbarPos("size", "Tracking")
    orig_ksize = cv.getTrackbarPos("orig_ksize", "Tracking")
    ksize = cv.getTrackbarPos("ksize", "Tracking")

    canny = cv.Canny(img2, 100, 200)

    if size%2 == 0:
        size += 1

    if orig_ksize%2 == 0:
        orig_ksize += 1

    if ksize%2 == 0:
        ksize += 1

    lap = cv.Laplacian(img2, cv.CV_64F, ksize=size)
    lap2 = np.uint8(np.absolute(lap))

    sobelX = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=size)
    sobelY = cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=size)

    sobelX2 = np.uint8(np.absolute(sobelX))
    sobelY2 = np.uint8(np.absolute(sobelY))

    sobelCombined = cv.bitwise_or(sobelX2, sobelY2)

    img4 = cv.cvtColor(sobelCombined, cv.COLOR_GRAY2BGR)

    img5 = cv.GaussianBlur(img, (orig_ksize, orig_ksize), 0)

    img6 = cv.GaussianBlur(img4, (ksize, ksize), 0)

    img7 = cv.bitwise_not(img4)

    img8 = cv.bilateralFilter(img7, 40, 90, 90)

    img9 = cv.subtract(img5, img6)

    img10 = cv.subtract(img6, img8)

    img11 = cv.bitwise_not(img10)

    img12 = cv.bitwise_and(img, img11)

    # cv.imshow("img", img)
    # cv.imshow("canny", canny)
    # cv.imshow("lap2", lap2)
    # cv.imshow("img3", img3)
    # cv.imshow("sobelX2", sobelX2)
    # cv.imshow("sobelY2", sobelY2)
    # cv.imshow("sobelCombined", sobelCombined)
    # cv.imshow("img4", img4)
    # cv.imshow("img5", img5)
    cv.imshow("img6", img6)
    # cv.imshow("img7", img7)
    cv.imshow("img8", img8)
    # cv.imshow("img9", img9)
    cv.imshow("img10", img10)
    cv.imshow("img11", img11)
    cv.imshow("img12", img12)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()

