import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


def colorCode(blue, green, red):
    return '#%02x%02x%02x' % (blue, green, red)


cv.namedWindow("Tracking")
cv.createTrackbar("blue", "Tracking", 0, 255, nothing)
cv.createTrackbar("green", "Tracking", 0, 255, nothing)
cv.createTrackbar("red", "Tracking", 0, 255, nothing)
cv.createTrackbar("white", "Tracking", 0, 255, nothing)

while True:
    blue = cv.getTrackbarPos("blue", "Tracking")
    green = cv.getTrackbarPos("green", "Tracking")
    red = cv.getTrackbarPos("red", "Tracking")
    white = cv.getTrackbarPos("white", "Tracking")

    b = np.ones((512, 512), dtype=np.uint8) * blue
    g = np.ones((512, 512), dtype=np.uint8) * green
    r = np.ones((512, 512), dtype=np.uint8) * red
    w = np.ones((512, 512), dtype=np.uint8) * white

    img = cv.merge((b, g, r))

    cv.putText(img, colorCode(blue, green, red), (100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

    # img1 = cv.merge((
    #     np.ones((512, 512), dtype=np.uint8) * 255,
    #     np.ones((512, 512), dtype=np.uint8) * 237,
    #     np.ones((512, 512), dtype=np.uint8) * 240
    # ))

    img1 = cv.merge((w, w, w))

    cv.putText(img, colorCode(white, white, white), (100, 300), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

    cv.imshow("img", img)
    cv.imshow("img1", img1)
    cv.imshow("add", cv.add(img, img1))
    cv.imshow("subtract", cv.subtract(img, img1))
    # cv.imshow("and", cv.bitwise_and(img, img1))
    # cv.imshow("or", cv.bitwise_or(img, img1))
    # cv.imshow("xor", cv.bitwise_xor(img, img1))
    # cv.imshow("not", cv.bitwise_not(img, img1))

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
