import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("1_2.jpg")

black = np.zeros((512, 512), dtype=np.uint8)
dark = cv.merge((black, black, black))

img1 = cv.bilateralFilter(img, 15, 90, 150)

# sobel testing

img2 = img1.copy()

img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sobelX_img2 = cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=1)
sobelY_img2 = cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=1)

process_sobelX_img2 = np.uint8(np.absolute(sobelX_img2))
process_sobelY_img2 = np.uint8(np.absolute(sobelY_img2))

sobelCombined_img2 = cv.bitwise_or(process_sobelX_img2, process_sobelY_img2)

img3 = cv.bitwise_not(sobelCombined_img2)

# cv.imshow("sobelCombined_img2", sobelCombined_img2)
# cv.imshow("img3", img3)
# cv.waitKey(0)
# cv.destroyAllWindows()

print(img3.shape)
print(img3)

img4 = img3.copy()

# for i in range(len(img3)):
#     for j in range(len(img3[i])):
#         if img3[i][j] < 220:
#             img4[i][j] = 0

img5 = cv.cvtColor(img3, cv.COLOR_GRAY2BGRA)

print(img5.shape)
print(img5[3])

b, g, r, a = cv.split(img5)

a1 = np.ones((512, 512), dtype=np.uint8) * 255

img6 = cv.merge((b, g, r, a1))

_, img7 = cv.threshold(img3, 180, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.waitKey(0)
# cv.destroyAllWindows()

# titles = ["sobelCombined_img2", "img3", "img4"]
# images = [sobelCombined_img2, img3, img4]

# for i in range(len(images)):
#     image = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
#     plt.subplot(2, 2, i+1)
#     plt.imshow(image, "gray")
#     plt.title(titles[i])
#
# plt.show()

img8 = cv.imread("2_2.jpg")

img9 = img[190:290, 190:250]
img10 = img8[190:290, 190:250]

img11 = img.copy()

cv.imshow("img11_copy", img11)

for i in range(len(img11)):
    for j in range(len(img11[i])):
        for k in range(len(img11[i][j])):
            if k == 0:
                if 90 < img11[i][j][k] < 190:
                    img11[i][j][k] += 30
            if k == 1:
                if 100 < img11[i][j][k] < 200:
                    img11[i][j][k] += 30
            if k == 2:
                if 170 < img11[i][j][k] < 220:
                    img11[i][j][k] += 30
                if 170 < img11[i][j][k] < 230:
                    img11[i][j][k] += 20
                if 170 < img11[i][j][k] < 240:
                    img11[i][j][k] += 10

cv.imshow("img11", img11)
# cv.imshow("img_add", cv.add(img, img))
# cv.imshow("img_add1", img + img)
# cv.imshow("img_add2", cv.addWeighted(img, 0.8, img, 0.5, 0))
cv.waitKey(0)
cv.destroyAllWindows()

print(img9.shape)

titles = ["img", "img8", "img9", "img10"]
images = [img, img8, img9, img10]

for i in range(len(images)):
    image = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    plt.subplot(2, 2, i+1)
    plt.imshow(image, "gray")
    plt.title(titles[i])

plt.show()
