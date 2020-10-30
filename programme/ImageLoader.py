"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: A class which will allow us to load the images for efficiently,
and effectively. This class will allow you to store all the paths for a given
path and will only load the images when needed instead of loading all the images
in memory at once.
"""

import os
import cv2 as cv
from Errors import *

class Image_Loader(object):
    """
    CLASS WAS ADAPTED FROM: Kuklin, Maxim. 2020. "Efficient image loading".
    Learn OpenCV.https://www.learnopencv.com/efficient-image-loading/
    """

    ext = (".png", ".jpg", ".jpeg")
    modes = ('RGB', 'HSV', 'LUV', 'BGR', 'LAB', 'GRAY')

    #choosing HSV as the defualt color channel as this typically gives better
    #results for image segementation
    def __init__(self, path, mode="HSV"):
        self._path = path
        self._mode = mode
        self._data = self.load(self.path)
        self._image_indx = 0

    @property
    def path(self):
        return self._path

    @property
    def mode(self):
        return self._mode

    @property
    def data(self):
        return self._data

    @path.setter
    def path(self, nw_path):
        self._path = nw_path
        #self._path = self.__validate_path(nw_path)
        #re-loading the data at the new path
        self._data = self.load(self.path)

    @mode.setter
    def mode(self, nw_mode):
        self._mode = self.__validate_mode(nw_mode)

    def load(self, path):
        """
        IMPORT: path (string)
        EXPORT: res (list)

        ASSERT: returns a list of image file[s]
        """
        if os.path.isfile(path):
            #res has to be the same data type as the return of load_dir so I
            #can access the images the same way regardless if it's a file or
            #a directory
            res = [path]

        if  os.path.isdir(path):
            res = self.load_dir(path)

        return res

    def load_dir(self, path):
        """
        IMPORT: path (string)
        EXPORT: paths (list)

        PURPOSE: returns a list of relitive paths of image files inside a
        specified directory
        """
        #getting all the images insde the folder as a list and returning it
        all_files = os.listdir(path)
        paths = [os.path.join(path, image) for image in all_files]

        return paths


    def load_image(self, path):
        """
        IMPORT: path (string)
        EXPORT: convert_img (numpy matrix)

        PURPPOSE: to load the image found in the specified path given the
        specified mode to load the image in
        """
        convert_img = cv.imread(path)
        if self._mode == 'GRAY':
            convert_img = cv.cvtColor(convert_img,cv.COLOR_BGR2GRAY)
        elif self._mode == 'HSV':
            convert_img = cv.cvtColor(convert_img,cv.COLOR_BGR2HSV)
        elif self._mode == 'LUV':
            convert_img = cv.cvtColor(convert_img,cv.COLOR_BGR2Luv)
        elif self._mode == 'LAB':
            convert_img = cv.cvtColor(convert_img,cv.COLOR_BGR2Lab)
        elif self._mode == "RGB":
            convert_img = cv.cvtColor(convert_img,cv.COLOR_BGR2RGB)

        return convert_img

    def create_labels(self):
        """
        EXPORT: labels (list)
        PURPOSE: to create labels from the loaded directories. Hence, it
        will make the labels the same as the directory names
        """
        labels = []
        for path in self._data:
            consititutes = path.split('/')
            #the name of the directory is always going to be the last index
            #of the splitted list
            labels.append(consititutes[-1])

        return labels


    def __iter__(self):
        """
        EXPORT: reference to this object

        PURPOSE: Defining how this class should be initialised when another function
        tries to iterate over this class
        """
        self._image_indx = 0
        return self

    def __len__(self):
        """
        EXPORT: the lenth of the data  class field
        PURPOSE: Defining how this class should behave when the user calls then
        len() function on this class
        """
        return len(self._data)

    def __next__(self):
        """
        EXPORT: the image matrix at the given index

        PURPOSE: Defining what this class should return and behave when the class
        is used in a for-each loop
        """
        if self._image_indx == len(self._data):
            raise StopIteration
        curr_img_path = self._data[self._image_indx]
        im = self.load_image(curr_img_path)

        self._image_indx += 1


        return im

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

    def __validate_mode(self, nw_mode):
        """
        IMPORT: nw_mode (string)
        EXPORT: nw_mode (string)
        PURPOSE: to validate if the new mode is a valid mode specified by the
        class
        """
        if nw_mode.upper().strip() not in self.modes:
            raise modeError("""
            mode type of:
            %s
            the mode must be one of these types:
            %s
            """ % (nw_mode, self.modes))
        return nw_mode
