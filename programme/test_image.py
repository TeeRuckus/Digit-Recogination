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
import os

class test_image(unittest.TestCase):
    im = cv.imread('../train_updated/tr17.jpg')
    test = Image(im, 1)
    def test_accessors(self):
        self.assertEqual(np.ndarray, type(self.test.im), "Image must be numpy"+
        " array")

    def test_setters(self):
        pass

    def test_debug(self):
        self.assertFalse(self.test.DEBUG, "initiliased debug = false")
        self.test.debug()
        self.assertTrue(self.test.DEBUG, 'toggled debug = True')
        self.test.debug()
        self.assertFalse(self.test.DEBUG, 'toggle debug back to false')

    def test_find_intersection(self):
        #TEST ONE: finding the intesection which will located on the
        #left side of the image
        boxes = np.array([[70,50,20,20], [110,20,20,20]], dtype='int32')
        box_one = boxes[0]
        box_two = boxes[1]
        intersect = (70, 20)
        res = self.test.find_intersection(box_one, box_two)[:2]

        self.assertEqual(intersect,res, "intersection on left side of"+
                " (70, 20)")

        #TEST TWO: finding the intersection which should be on the right
        #side of the image
        intersect = (130, 70)
        res = self.test.find_intersection(box_one, box_two, True)
        for indx, cord in enumerate(intersect):
            self.assertEqual(cord, res[indx] + res[indx + 2], "intersection "+
                    "on right side of")

    def test_finding_left_most_pt(self):
        #CASE 1: they is a clear left most box
        bboxes = np.array([[20, 50, 20, 20], [60, 80, 20,20]], dtype='int32')
        left_most_pt = (20, 50)
        found_pt = self.test.find_leftmost_pt(bboxes)[:2]
        self.assertEqual(left_most_pt, found_pt, 'CASE 1: left most point'+
        ' (20,50)')

        #CASE 2: they is a clear right most box
        right_most_pt = (80, 100)
        found_pt  = self.test.find_leftmost_pt(bboxes, True)
        for  indx, cord in enumerate(right_most_pt):
            self.assertEqual(cord, found_pt[indx] + found_pt[indx+2],
                    "CASE 2: right most point (80, 100)")

        #CASES 3: boxes are at the same x-axis but need to ensure that the
        #upper most box is going to be selected
        bboxes = np.array([[20, 50, 20, 20], [20, 100, 20,20]], dtype='int32')
        found_pt = self.test.find_leftmost_pt(bboxes)[:2]
        self.assertEqual(left_most_pt, found_pt, 'case 2: ensuring left most'+
                " upper most box is going to be selected")

        #CASE 4: same as case 3 but on the right side and it needs to be
        #lower most box which is going to be selected
        bboxes = np.array([[50, 20, 20, 20], [110,50,20,20]])
        right_most_pt = (130, 70)
        found_pt = self.test.find_leftmost_pt(bboxes, True)

        for indx, cord in enumerate(right_most_pt):
            res_pnt = found_pt[indx] + found_pt[indx + 2]
            self.assertEqual(cord, res_pnt, "the right most point is used"+
            "but, the box is not the lowest box hence the intersection is "+
            "formed of (130, 70)")

        #CASE 5: left_most box selected is not the upper most box, hence,
        #where the eges of the boxes intersect should be selected
        bboxes = np.array([[70,50,20,20], [90,25,20,20]], dtype='int32')
        intersect = (70, 25)
        found_intersect = self.test.find_leftmost_pt(bboxes)[:2]
        self.assertEqual(intersect, found_intersect, "CASE 5: left most found"+
        "boxfound | not highest | intersect @ (70, 25)")

        bboxes = np.array([[110,25,20,20], [90,50, 20,20]], dtype='int32')
        intersect = (130,70)
        found_intersect = self.test.find_leftmost_pt(bboxes, True)
        print('intersection ', found_intersect)
        for indx, cord in enumerate(intersect):
            self.assertEqual(cord,found_intersect[indx] +
                    found_intersect[indx+2], "found right most point | not"+
                    "lowest point | formed intersect @ (130, 70")


        #CASE 6: they is  a box inside a box
    def test_get_ROI(self):
        #toggling the debugging feature on
        self.test.debug()
        all_imgs = os.listdir('../train_updated/')
        #interest = ['tr05.jpg', 'tr09.jpg']
        #interest = ['tr08.jpg', 'tr25.jpg']
        #interest = ['tr09.jpg', 'tr12.jpg', 'tr03.jpg', 'tr08.jpg', 'tr02.jpg']
        interest = ['tr04.jpg']
        for img_id, path in enumerate(all_imgs):
            print(yellow + 'path'  + reset, path)
            im = cv.imread("../train_updated/" + path)
            digits = self.test.get_ROI(im,img_id)[1]
            for num, digit in enumerate(digits):
                cv.imshow("digit: %s" %num, digit)
            cv.waitKey()
            cv.destroyAllWindows()
