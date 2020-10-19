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

    @property
    def im(self):
        return self._im

    @im.setter
    def im(self, in_im):
        self._im = self.get_interest_area(in_im)

    def get_interest_area(self, im):
        return im

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
