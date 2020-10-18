"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: is to created labelled data set given a directory
name
"""
from ImageLoader import *

class Data_Set(object):
    """
    CLASSFIELDS:

    PURPOSE: Over arching class which will contain each and every single
    individual set
    """

    class __set(object):
        """
        CLASSFIELDS:

        PURPOSE: private inner class which is the respective data set
        found in the over arching data set
        """
        def __init__(self, label, set_dir, mode='HSV'):
            #I don't need to make these fields private because it's a
            #private inner class Data_set is the only class which can
            #control this class
            self.label = label
            self.data = self.create_set(set_dir, mode)
            self.set_dir = set_dir
            self.mode = mode

        def create_set(self, set_dir, mode):
            im_set = Image_Loader(set_dir, mode)

    def __init__(self, set_path):
        self._mode = 'HSV'
        self._set = self.create_set(set_path)
        self._set_path = set_path
        self._no_set = len(self._set)

    @property
    def set(self):
        return self._set

    @property
    def set_path(self):
        return self._set_path

    @property
    def no_set(self):
        return len(self._set)

    @property
    def mode(self):
        return self._mode

    @set.setter
    def set(self, nw_set_dir):
        self._set = self.create_set(nw_set_dir)
        self._set_path = nw_set_dir

    @mode.setter
    def mode(self, nw_mode):
        self._mode = nw_mode
        self._set = create_set(nw_set_dir, self._mode)


    def create_set(self, in_path):
        """
        IMPORT:
        EXPORT:

        PURPOSE:
        """
        loaded_dir  = Image_Loader(in_path, self._mode)
        labels = loaded_dir.create_labels()
        data_set = dict()

        #creating all the set
        for label, set_dir in zip(labels, loaded_dir.data):
            data_set[label] = self.__set(label, set_dir, self._mode)

        return data_set


