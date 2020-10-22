import cv2 as cv
import numpy as np

white_img = 255 * np.ones(36)
white_img = white_img.reshape(6, 6)

cv.imshow('white image', white_img)
cv.waitKey()

print(white_img)
