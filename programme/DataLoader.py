"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE
"""

import os

class Image_Loader(object):
    """
    PURPOSE OF CLASS

    CLASS WAS ADAPTED FROM: Kuklin, Maxim. 2020. "Efficient image loading".
    Learn OpenCV.https://www.learnopencv.com/efficient-image-loading/
    """

    ext = (".png", ".jpg", ".jpeg")

    def __init__(self, path, mode="BGR"):
        self._path = path
        self._mode = mode

    @property
    def path(self):
        return self._path

    @property
    def mode(self):
        return self._mode

    @path.setter
    def path(self, nw_path):
        self._path = __validate_path(nw_path)


    def __validate_path(self, nw_path):
        if (nw_path is None or nw_path == "")  and not(isinstance(nw_path,str)):
            raise PathError('INVALID: Path Error')

        return nw_path
