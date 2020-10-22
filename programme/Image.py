"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - have the trainer to be able to train against occulisions in the image
"""

from abc import abstractmethod

class Image(object):
    """
    """
    def __init__(self, im):
       self._im = self.get_interest_area(im)
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

    def get_interest_area(self, im):
        return im

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

        return nw_x, nw_y, nw_w, nw_h

    #AUGMENTATION OPERATION METHODS
    def rotate_im(self):
        """
        """

    def scale_im(self):
        """
        """

    def bias_im(self):
        """
        """

    def gain_im(self):
        """
        """

    def equalize_hists(self):
        """
        """

    #FEATURE DECRIPTOR METHODS
    def SIFT(self):

        """
        """

    def harris(self, thresh):
        """
        """

    def hog_des(self, im):
        """
        """
