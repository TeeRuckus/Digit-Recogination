"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: is to created labelled data set given a directory
name, this data structure is what is going to be used to do trainning on
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
            self.data = self.create_set(set_dir, mode)
            self.set_dir = set_dir
            self.mode = mode

        def create_set(self, set_dir, mode):
            """
            IMPORT: set_dir: string, mode: string
            EXPORT: Imager_Loader object

            PURPOSE: to  crete an imaeg loader set which will contain the paths
            of image files inside a directory
            """
            return Image_Loader(set_dir, mode)

    def __init__(self, set_path):
        self._mode = 'HSV'
        #this is going to be a list whereby the 0 index is the actual set,
        #and the 1 index is going to be the index
        self._set = self.create_set(set_path)
        self._set_path = set_path
        self._no_set = len(self._set[0])

    #ACCESORS
    @property
    def set(self):
        return self._set[0]

    @property
    def labels(self):
        return self._set[1]

    @property
    def set_path(self):
        return self._set_path

    @property
    def no_set(self):
        return len(self._set[0])

    @property
    def mode(self):
        return self._mode

    #MUTATORS
    @set.setter
    def set(self, nw_set_dir):
        nw_set_dir = self.__validate_path(nw_set_dir)
        self._set = self.create_set(nw_set_dir)
        self._set_path = nw_set_dir

    @mode.setter
    def mode(self, nw_mode):
        self._mode = nw_mode
        self._set = create_set(nw_set_dir, self._mode)


    def create_set(self, in_path):
        """
        IMPORT: in_path: string
        EXPORT: data_set: hash table

        PURPOSE: to convert the image loader of direcories into their each
        respective set of images which are stored in a hash table with the
        directory name as a key
        """
        loaded_dir  = Image_Loader(in_path, self._mode)
        labels = loaded_dir.create_labels()
        data_set = dict()

        #creating all the set
        for label, set_dir in zip(labels, loaded_dir.data):
            data_set[label] = self.__set(label, set_dir, self._mode)

        return data_set, labels

    def __validate_path(self, nw_path):
        """
        IMPORT: nw_path (string)
        EXPORT: nw_path (string)

        PURPOSE: to validate if the passed nw_path string is not an empty
        string and it's a string, to ensure that the path is an actual image
        file or it's a directory
        """
        if (nw_path is None or nw_path == "")  and not(isinstance(nw_path,str)):
            raise PathError("""
            path path can't be an empty string or not a string:
            input:
            {}
            type:
            {}""".format(nw_path, type(nw_path)))

        if not (os.path.isfile(nw_path) or os.path.isdir(nw_path)):
            raise PathError("path is not a valid file or directory: %s"
                    %nw_path)

        if (os.path.isfile(nw_path)):
            if not nw_path.lower.endswith(self.ext):
                raise PathError("not recognised file extension %s" % nw_path)

        return nw_path


