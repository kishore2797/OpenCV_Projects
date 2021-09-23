import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

white = np.ones((512, 512), dtype=np.uint8) * 255
light = cv.merge((white, white, white))

black = np.zeros((512, 512), dtype=np.uint8)
dark = cv.merge((black, black, black))


def nothing(x):
    pass


cv.namedWindow("Tracking")
cv.createTrackbar("alpha_light", "Tracking", 0, 10, nothing)
cv.createTrackbar("beta_light", "Tracking", 0, 10, nothing)
cv.createTrackbar("img_ksize", "Tracking", 0, 60, nothing)
cv.createTrackbar("img_light_ksize", "Tracking", 0, 60, nothing)
cv.createTrackbar("img_dark_ksize", "Tracking", 0, 60, nothing)

while True:
    alpha_light = cv.getTrackbarPos("alpha_light", "Tracking") / 10
    beta_light = cv.getTrackbarPos("beta_light", "Tracking") / 10
    img_ksize = cv.getTrackbarPos("img_ksize", "Tracking")
    img_light_ksize = cv.getTrackbarPos("img_light_ksize", "Tracking")
    img_dark_ksize = cv.getTrackbarPos("img_dark_ksize", "Tracking")

    if img_ksize%2 == 0:
        img_ksize += 1

    if img_light_ksize%2 == 0:
        img_light_ksize += 1

    if img_dark_ksize%2 == 0:
        img_dark_ksize += 1

    img_blur = cv.GaussianBlur(img, (img_ksize, img_ksize), 0)

    img_light = cv.addWeighted(img_blur, alpha_light, light, beta_light, 0.0)
    img_dark = cv.addWeighted(img_blur, alpha_light, dark, 0.0, 0.0)

    img_light_blur = cv.GaussianBlur(img_light, (img_light_ksize, img_light_ksize), 0)
    img_dark_blur = cv.GaussianBlur(img_dark, (img_dark_ksize, img_dark_ksize), 0)

    img_light_dark_blur_add = cv.add(img_light_blur, img_dark_blur)

    img_add = cv.add(img_blur, img_light_dark_blur_add)

    cv.imshow("img", img)
    cv.imshow("img_light", img_light)
    cv.imshow("img_dark", img_dark)
    cv.imshow("img_light_blur", img_light_blur)
    cv.imshow("img_dark_blur", img_dark_blur)
    cv.imshow("img_light_dark_blur_add", img_light_dark_blur_add)
    cv.imshow("img_add", img_add)
    cv.imshow("img_sub", cv.subtract(img_add, img_light_blur))

    key = cv.waitKey(1)
    if key == 27:
        break


img1 = cv.bilateralFilter(img, 15, 100, 180)

img2 = cv.subtract(img, img1)

img3 = cv.addWeighted(img1, 1.0, img2, 1.0, 0)

img4 = cv.bitwise_not(img2)

img5 = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)

img6 = cv.adaptiveThreshold(img5, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 4)

img7 = cv.cvtColor(img6, cv.COLOR_GRAY2BGR)

img8 = cv.addWeighted(img3, 1.0, img7, 1.0, 0)

img9 = cv.cvtColor(img8, cv.COLOR_BGR2GRAY)

img10 = cv.cvtColor(img9, cv.COLOR_GRAY2BGR)

img11 = cv.bitwise_and(img3, img10)

img12 = cv.bitwise_and(img3, img7)

# sobel testing

img13 = img1.copy()

img13 = cv.cvtColor(img13, cv.COLOR_BGR2GRAY)

sobelX_img = cv.Sobel(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.CV_64F, 0, 1, ksize=3)
sobelX_img13 = cv.Sobel(img13, cv.CV_64F, 0, 1, ksize=3)

process_sobelX_img = np.uint8(np.absolute(sobelX_img))
process_sobelX_img13 = np.uint8(np.absolute(sobelX_img13))

# cv.imshow("img1", img1)
# cv.imshow("sobelX_img", sobelX_img)
# cv.imshow("process_sobelX_img", process_sobelX_img)
# cv.imshow("process_sobelX_img13", process_sobelX_img13)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img6", img6)
# cv.imshow("img8", img8)
# cv.imshow("img10", img10)
# cv.imshow("img11", img11)
# cv.imshow("img12", img12)
# cv.waitKey(0)
cv.destroyAllWindows()