import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

black = np.zeros((512, 512), dtype=np.uint8)
dark = cv.merge((black, black, black))

img1 = cv.bilateralFilter(img, 15, 90, 150)

# sobel testing

img2 = img1.copy()

img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sobelX_img = cv.Sobel(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.CV_64F, 0, 1, ksize=3)
process_sobelX_img = np.uint8(np.absolute(sobelX_img))

sobelX_img2 = cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=1)
sobelY_img2 = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=1)

process_sobelX_img2 = np.uint8(np.absolute(sobelX_img2))
process_sobelY_img2 = np.uint8(np.absolute(sobelY_img2))

sobelCombined_img2 = cv.bitwise_or(process_sobelX_img2, process_sobelY_img2)

img3 = cv.adaptiveThreshold(process_sobelX_img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 15)
img4 = cv.Canny(process_sobelX_img2, 100, 200)

img5 = cv.bitwise_not(process_sobelX_img2)

img6 = cv.cvtColor(img5, cv.COLOR_GRAY2BGR)
img7 = cv.cvtColor(img6, cv.COLOR_BGR2HSV)

img8 = cv.GaussianBlur(img6, (5, 5), 0)
img9 = cv.medianBlur(img6, 5)

img10 = cv.addWeighted(img8, 0.8, dark, 1.0, 0)
img11 = cv.addWeighted(img9, 0.8, dark, 1.0, 0)

_, img12 = cv.threshold(img5, 127, 255, cv.THRESH_BINARY)

img13 = cv.bitwise_not(sobelCombined_img2)

startValue = 150

_, img14 = cv.threshold(sobelCombined_img2, startValue, 255, cv.THRESH_BINARY)
_, img15 = cv.threshold(img13, startValue, 255, cv.THRESH_BINARY)

img16 = cv.cvtColor(img13, cv.COLOR_GRAY2BGR)
img17 = cv.cvtColor(img16, cv.COLOR_BGR2HSV)

_, img18 = cv.threshold(sobelCombined_img2, startValue, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

img19 = cv.cvtColor(img18, cv.COLOR_GRAY2BGR)
img20 = cv.addWeighted(img1, 1.0, img19, 1.0, 0)

img21 = cv.GaussianBlur(img19, (5, 5), 0)

img22 = cv.divide(img1, img21, scale=256)

# img23 = cv.bitwise_and(img22, img1)

img24 = cv.cvtColor(img22, cv.COLOR_BGR2GRAY)

img25 = cv.GaussianBlur(img18, (5, 5), 0)

img26 = img24 / img25

img27 = cv.multiply(img24, img25, scale=1/256)

img28 = cv.cvtColor(img27, cv.COLOR_GRAY2BGR)

img29 = cv.addWeighted(img28, 0.8, img1, 0.6, 0)

img30 = cv.addWeighted(img29, 0.8, dark, 0.0, 0)

img31 = cv.GaussianBlur(img30, (5, 5), 0)

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("sobelX_img", sobelX_img)
# cv.imshow("process_sobelX_img", process_sobelX_img)
# cv.imshow("process_sobelX_img2", process_sobelX_img2)
# cv.imshow("process_sobelY_img2", process_sobelY_img2)
# cv.imshow("sobelCombined_img2", sobelCombined_img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.imshow("img8", img8)
# cv.imshow("img9", img9)
# cv.imshow("img10", img10)
# cv.imshow("img11", img11)
# cv.imshow("img12", img12)
# cv.imshow("img13", img13)
# cv.imshow("img14", img14)
# cv.imshow("img15", img15)
# cv.imshow("img17", img17)
# cv.imshow("img18", img18)
# cv.imshow("img19", img19)
# cv.imshow("img20", img20)
cv.imshow("img21", img21)
cv.imshow("img22", img22)
cv.imshow("img24", img24)
cv.imshow("img25", img25)
cv.imshow("img26", img26)
cv.imshow("img27", img27)
cv.imshow("img29", img29)
cv.imshow("img30", img30)
cv.imshow("img31", img31)
cv.waitKey(0)
cv.destroyAllWindows()


def nothing(x):
    pass


# cv.namedWindow("Tracking")
# cv.createTrackbar("LH", "Tracking", 0, 255, nothing)  # LH -> Lower Hue
# cv.createTrackbar("LS", "Tracking", 0, 255, nothing)  # LS -> Lower Saturation
# cv.createTrackbar("LV", "Tracking", 0, 255, nothing)  # LV -> Lower Value
# cv.createTrackbar("UH", "Tracking", 255, 255, nothing)  # UH -> Upper Hue
# cv.createTrackbar("US", "Tracking", 255, 255, nothing)  # US -> Upper Saturation
# cv.createTrackbar("UV", "Tracking", 255, 255, nothing)  # UV -> Upper Value
#
# while True:
#     l_h = cv.getTrackbarPos("LH", "Tracking")
#     l_s = cv.getTrackbarPos("LS", "Tracking")
#     l_v = cv.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv.getTrackbarPos("UH", "Tracking")
#     u_s = cv.getTrackbarPos("US", "Tracking")
#     u_v = cv.getTrackbarPos("UV", "Tracking")
#
#     lower_bound = np.array([l_h, l_s, l_v], np.uint8)
#     upper_bound = np.array([u_h, u_s, u_v], np.uint8)
#
#     mask = cv.inRange(img17, lower_bound, upper_bound)
#
#     res = cv.bitwise_and(img1, img1, mask=mask)
#     res1 = cv.bitwise_and(img, img, mask=mask)
#
#     res2 = cv.add(img, img)
#
#     res3 = cv.medianBlur(res2, 5)
#
#     res4 = cv.bitwise_and(res3, res1)
#
#     cv.imshow("mask", mask)
#     cv.imshow("res", res)
#     cv.imshow("res1", res1)
#     cv.imshow("res2", res2)
#     cv.imshow("res3", res3)
#     cv.imshow("res4", res4)
#     cv.imshow("res5", cv.add(res1, cv.addWeighted(res3, .4,
#                              cv.cvtColor(cv.bitwise_not(cv.cvtColor(res4, cv.COLOR_BGR2GRAY)),
#                                          cv.COLOR_GRAY2BGR), 0.2, 0)))
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break