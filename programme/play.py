import cv2 as cv
import numpy as np
import  numpy.ma as ma
from Image import Image


bboxes = [[-1, -1, -1, -1], [-1, -1, -1, -1], [9,5,8,6], [10, 20, 30, 40],
        [100,20,30, 40]]




del bboxes[0]
del bboxes[0]


print(bboxes)
bboxes = np.array(bboxes, dtype='int32')
print(bboxes.mean())
