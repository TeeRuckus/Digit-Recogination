"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE: to  test the class of DataSet to enusre that the functionality
meets the expected functionality

TO DO:
"""

import unittest
import cv2 as cv
import numpy as np
from Image import *

class test_image(unittest.TestCase):
    im = cv.imread('../train_updated/tr06.jpg')
    test = Image(im)
    shape = (220, 220, 3)
    def test_accessors(self):
        self.assertEqual(np.ndarray, type(self.test.im), "Image must be numpy"+
        " array")
        self.assertEqual(self.shape, self.test.im.shape, "ensuring that this"+
        " is an actual image given")


    def test_setters(self):
        im = cv.imread('../val_updated/val04.jpg')
        self.test.im = im
        nw_size= (2430, 1980, 3)

        self.assertEqual(nw_size, self.test.im.shape, "has the image updated"+
                " to the new image")

