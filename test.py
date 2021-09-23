import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

img3 = cv.imread("2_2.jpg")

# cv.imshow("img2", cv.xphoto.inpaint(img, img3,None, cv.xphoto.INPAINT_FSR_BEST))

img_sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, 3)
img_sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, 3)

img_sobelX = np.uint8(np.absolute(img_sobelX))
img_sobelY = np.uint8(np.absolute(img_sobelY))

img_sobelCombined = cv.bitwise_or(img_sobelX, img_sobelY)

img4 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img4_sobelX = cv.Sobel(img4, cv.CV_64F, 1, 0, 3)
img4_sobelY = cv.Sobel(img4, cv.CV_64F, 0, 1, 3)

img4_sobelX = np.uint8(np.absolute(img4_sobelX))
img4_sobelY = np.uint8(np.absolute(img4_sobelY))

img4_sobelCombined = cv.bitwise_or(img4_sobelX, img4_sobelY)

img5 = cv.bilateralFilter(img4, 9, 80, 80)

img5_sobelX = cv.Sobel(img5, cv.CV_64F, 1, 0, 2)
img5_sobelY = cv.Sobel(img5, cv.CV_64F, 0, 1, 2)

img5_sobelX = np.uint8(np.absolute(img5_sobelX))
img5_sobelY = np.uint8(np.absolute(img5_sobelY))

img5_sobelCombined = cv.bitwise_or(img5_sobelX, img5_sobelY)

img6 = cv.bitwise_not(img5_sobelCombined)

img7 = cv.cvtColor(img6, cv.COLOR_GRAY2BGR)

img9 = cv.adaptiveThreshold(img4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 6)

img11 = cv.cvtColor(img9, cv.COLOR_GRAY2BGR)

# cv.imshow("img_sobelCombined", img_sobelCombined)
# cv.imshow("img4_sobelCombined", img4_sobelCombined)
# cv.imshow("img5", img5)
# cv.imshow("img5_sobelCombined", img5_sobelCombined)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)

def nothing(x):
    pass


cv.namedWindow("Tracking")
cv.createTrackbar("size", "Tracking", 1, 20, nothing)
cv.createTrackbar("dynRatio", "Tracking", 1, 100, nothing)
cv.createTrackbar("LH", "Tracking", 0, 255, nothing)  # LH -> Lower Hue
cv.createTrackbar("LS", "Tracking", 0, 255, nothing)  # LS -> Lower Saturation
cv.createTrackbar("LV", "Tracking", 0, 255, nothing)  # LV -> Lower Value
cv.createTrackbar("UH", "Tracking", 255, 255, nothing)  # UH -> Upper Hue
cv.createTrackbar("US", "Tracking", 255, 255, nothing)  # US -> Upper Saturation
cv.createTrackbar("UV", "Tracking", 255, 255, nothing)  # UV -> Upper Value

while True:
    size = cv.getTrackbarPos("size", "Tracking")
    dynRatio = cv.getTrackbarPos("dynRatio", "Tracking")

    l_h = cv.getTrackbarPos("LH", "Tracking")
    l_s = cv.getTrackbarPos("LS", "Tracking")
    l_v = cv.getTrackbarPos("LV", "Tracking")

    u_h = cv.getTrackbarPos("UH", "Tracking")
    u_s = cv.getTrackbarPos("US", "Tracking")
    u_v = cv.getTrackbarPos("UV", "Tracking")

    lower_bound = np.array([l_h, l_s, l_v], np.uint8)
    upper_bound = np.array([u_h, u_s, u_v], np.uint8)

    img1 = cv.xphoto.oilPainting(img, size, dynRatio, cv.COLOR_BGR2Lab)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv_img, lower_bound, upper_bound)

    res = cv.bitwise_and(img1, img1, mask=mask)

    img7 = cv.bitwise_and(img1, img7)

    img10 = cv.bitwise_and(img11, img1)

    

    # cv.xphoto.inpaint(img1, img5, img7, cv.xphoto.INPAINT_FSR_FAST)

    # cv.imshow("img7", img7)
    # cv.imshow("img1", img1)
    # cv.imshow("res", res)
    cv.imshow("img10", img10)
    cv.imshow("img11", img11)

    key = cv.waitKey(1)

    if key == 27:
        break

cv.destroyAllWindows()