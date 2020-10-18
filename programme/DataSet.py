"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: is to created labelled data sets given a directory
name
"""
from ImageLoader import *

class Data_Set(object):
    """
    PURPOSE: Over arching class which will contain each and every single
    individual set
    """

    class __set(object):
        """
        PURPOSE: private inner class which is the respective data set
        found in the over arching data set
        """
        def __init__(self, label, set_dir, mode='HSV'):
            #I don't need to make these fields private because it's a
            #private inner class Data_set is the only class which can
            #control this class
            self.label = label
            self.data = create_set(set_dir, mode)
            self.set_dir = set_dir
            self.mode = mode

        def create_set(self, set_dir, mode):
            im_set = Image_Loader(set_dir, mode)

    def __init__(self, sets_path):
        self._sets = create_sets(sets_path)
        self._sets_path = sets_path
        self._no_sets = len(self.sets)

    @property
    def sets(self):
        return self._sets

    @property
    def sets_path(self):
        return self._sets_path

    @property
    def no_sets(self):
        return self._no_sets

    @sets.setter
    def sets(self, nw_set_dir, mode='HSV'):
        self._sets = create_sets(nw_set_dir, mode)

