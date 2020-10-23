"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - have the trainer to be able to train against occulisions in the image
"""

from abc import abstractmethod
import numpy as np
from Errors import *
from Colours import *
import cv2 as cv

class Image(object):
    """
    """
    def __init__(self, im):
       self._im = self.get_ROI(im)
       #I don't know how to use unittest with images :(, so I am going to build
       #the debugging into this module
       self._DEBUG = False

    @property
    def im(self):
        return self._im

    @property
    def DEBUG(self):
        return self._DEBUG

    @im.setter
    def im(self, in_im):
        self._im = self.get_interest_area(in_im)

    def debug(self):
        """
        PURPOSE: to act as a toggle to witch the debugging features for this
        class
        """
        if self.DEBUG:
            self._DEBUG = False
        else:
            self._DEBUG = True

    def get_ROI(self, im, **kwargs):
        im = self._validate_image(im)

        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        #decreasing the required memory the image needs but still keeping the
        #important features
        gray = cv.GaussianBlur(gray, (5,5), 0)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

        edge_thresh = 100
        #the openCV doc recommends that you will have your upper thresh hold
        #twice as the lower threshold
        canny_trans = cv.canny(thresh, edge_thresh, edge_thresh * 2)
        rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        canny_trans = cv.erode(thresh, None, iterations=1)
        canny_trans = cv.dilate(thresh, None, iterations=1)
        canny_trans_invert = canny_trans.max() - canny_trans

    #FUNCTIONS WHICH WILL HELP TO FIND THE INTEREST AREA

    def find_leftmost_pt(self, bboxes, reverse=False):
        """
        if reverse is true, it will find the right-most lower most point of the
        bounding boxes
        """
        left_most_boxes = sorted(bboxes, key=lambda x: x[0], reverse=reverse)

        temp_box = left_most_boxes[0]

        #CASE 1: clear left most box will be met if it fails CASE 2's and
        #CASE 3's checks

        #CASE 2: boxes are the same x-coordinate hence to enusre that
        #the upper-most box is selected
        for box in left_most_boxes:
            #case: when two boxes have the same x-dimension but differing
            #y-dimensions
            if temp_box[0] == box[0]:
                if temp_box[1] > box[1]:
                    temp_box = box

        #CASE 3: the left most box is selected but if they's a box which is
        #higher than the current box combine find the intersecting points
        highest_boxes = sorted(bboxes, key=lambda y: y[1], reverse=reverse)
        highest_box = highest_boxes[0]

        equal = highest_box == temp_box
        #if the current box is not the highest box, form an intersection
        #with the highest box
        if not equal.all():
            temp_box = self.find_intersection(highest_box, temp_box,
                    reverse=reverse)

        if self._DEBUG:
            print('='*80)
            print(red, 'find_leftmost_p() | temp box',reset)
            print('\t {}'.format(temp_box))
            print('='*80)

        return temp_box[0], temp_box[1], temp_box[2], temp_box[3]

    def find_intersection(self, box_one, box_two, reverse=False):
        """
        IMPORT:
        EXPORT:

        PURPOSE
            if you're trying to find the right most corner, set reverse to
            true, and add the width, and the height of the object of interset
        """
        temp_boxes = [box_one, box_two]
        #placing the box with the lowest x value at the front
        temp_boxes = sorted(temp_boxes, key=lambda x: x[0], reverse=reverse)
        #the first boxes x coordinate
        nw_x = temp_boxes[0][0]
        #the right most point will be the temp box's in reverse, and it
        #will be the that boxes x value plus that boxes w value
        nw_w = temp_boxes[0][2]
        #placing the box withthe lowest y value at the front
        temp_boxes = sorted(temp_boxes, key=lambda  y: y[1], reverse=reverse)
        #the first boxes y coordinate
        nw_y = temp_boxes[0][1]
        #the right most point will be the temp boxes in reverse, and it
        #will be that  box's y value plus box's h value
        nw_h = temp_boxes[0][3]

        if self._DEBUG:
            print('='* 80)
            print(red, 'find_intersection() | interesction',reset)
            print('\t {}, {}, {}, {}'.format(nw_x, nw_y, nw_w, nw_h))
            print('='* 80)

        return nw_x, nw_y, nw_w, nw_h

    def draw_boxes(self, bboxes, im, color=(0,0,255)):
        """
        IMPORT:
        EXPORT:

        PURPOSE: it's to draw a bounding box which is either a shape of a
        rectangle or a square
        """
        for box in bboxes:
            #if the box has been labelled by a negative -1 by a filtering
            #algorithm we should skip over this box
            if box[0] == -1:
                pass
            else:
                x,y,w,h = box
                cv.rectangle(im, (x,y), (x+w, y+h), color, 2)


    def find_clusters(self, bboxes):
        """
        YOU HAVEN'T TESTED THIS AS YET
        """
        cluster = []

        bboxes =  sorted(bboxes, key=lambda x: x[0])

        for curr_box in bboxes:
            if curr_box[0] == -1:
                pass
            else:
                x,y,w,h = curr_box
                pt1 = (x, y)
                pt2 = (x+w, y+h)
                for alt_box in bboxes:
                    if alt_box[0] == -1:
                        pass
                    else:
                        x_alt,y_alt,w_alt,h_alt = alt_box
                        pt1_alt = (x_alt,y_alt)
                        pt2_alt =(x_alt+w_alt, y_alt+h_alt)

                        x_diff = abs(pt2[0] - pt1_alt[0])
                        #y_diff = abs(pt1[0] - pt2_alt[1])
                        y_diff = abs(pt2[1] - pt1_alt[1])

                        #YOU'RE NOT COMPARING THE LENGTHS OF THE LINES HERE, YOU'RE
                        #COMPARING THE POINTS, THAT'S MAYBE WHY YOUR POINTS ARE SHIT
                        line_seg_x = max(pt2[0], pt2_alt[0])
                        line_seg_y = max(pt2[1], pt2_alt[1])

                        line_TOL_x  = line_seg_x * 0.15
                        line_TOL_y = line_seg_y * 0.15

                        line_TOL_x = 0
                        line_TOL_y = 0

                        if x_diff < pt2[0]  and x_diff < pt2_alt[0]:
                                if y_diff < h:
                                    cluster.append([curr_box, alt_box])


    def validate_image(self, in_im):
        #an image is going to be an numpy matrice
        if not type(in_im) == np.ndarry:
            #all loaded images, are an unsigned interger by defualt
            if not in_im.dtype == 'uint8':
                raise ImageError("Error: an image wasn't laoded in the system")

        return in_im
