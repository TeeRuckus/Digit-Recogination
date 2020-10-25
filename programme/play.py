import cv2 as cv
import numpy as np
import  numpy.ma as ma
from Image import Image


bboxes = [[-1, -1, -1, -1], [-1, -1, -1, -1], [9,5,8,6], [10, 20, 30, 40]]


mask = ma.masked_equal(bboxes, -1)
print(mask)
mask[~np.array(mask)]
