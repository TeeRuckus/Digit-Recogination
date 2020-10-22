import cv2 as cv
import numpy as np

white_img = 127 * np.ones(400, dtype='uint8')
white_img = white_img.reshape(20,20)
cv.rectangle(white_img, (0,10), (10,20), (255,0,0))

cv.imshow('white image', white_img)
cv.waitKey()

