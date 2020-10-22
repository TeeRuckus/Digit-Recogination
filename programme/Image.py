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
            self.DEBUG == False
        else:
            self.DEBUG == True

    def get_interest_area(self, im):
        return im

    #FUNCTIONS WHICH WILL HELP TO FIND THE INTEREST AREA

    def find_leftmost_point(self, bboxes, reverse=False):
        """
        if reverse is true, it will find the right-most lower most point of the
        bounding boxes
        """
        left_most_boxes = sorted(bboxes, key=lambda x: x[0], reverse=reverse)

        temp_box = left_most_boxes[0]

        #ensuring we're going to grab the upper-most box
        for box in left_most_boxes:
            #case: when two boxes have the same x-dimension but differing
            #y-dimensions
            if temp_box[0] == box[0]:
                if temp_box[1] < box[1]:
                    temp_box = box
    return temp_box[0], temp_box[1]

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
