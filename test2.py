import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

img1 = cv.imread("mask1.png")

img1 = cv.resize(img1, (512, 512))

img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

_, img2 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

img3 = img.copy()

img3[np.where(img2 == 255)] = img2[np.where(img2 == 255)]

img4 = img * 0.5 + img2 * 0.5

img5 = img4.copy()

img4 = img4.astype(np.uint8)

img5 = img5 + (96, 128, 160)

img5 = img5.clip(0, 255)

img5 = img5.astype(np.uint8)

img6 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)

img7 = img6 / 255

img7 = img * img7

img7 = img7.astype(np.uint8)

# img8 = cv.fastNlMeansDenoisingColored(img, None, 20, 5, 19, 3)
#
# img9 = cv.add(img8, img8)

# img10 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
# img11 = cv.medianBlur(cv.cvtColor(img10, cv.COLOR_RGB2GRAY), 3)
#
# circles = cv.HoughCircles(img11, cv.HOUGH_GRADIENT, 5, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
#
# circles = np.uint16(np.round(circles))
#
# masking = np.full((img10.shape[0], img10.shape[1]), 0, dtype=np.uint8)
#
# for j in circles[0, :]:
#     cv.circle(masking, (j[0], j[1]), j[2], (255, 255, 255), -1)
#
# final_img = cv.bitwise_or(img10, img10, mask=masking)
#
# final_img = cv.cvtColor(final_img, cv.COLOR_RGB2BGR)

# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
cv.imshow("img7", img7)
# cv.imshow("img8", img8)
# cv.imshow("img9", img9)
# cv.imshow("final_img", final_img)
cv.waitKey(0)
cv.destroyAllWindows()