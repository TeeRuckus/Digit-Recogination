"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: it's to define custom exception. So it allows the user to be
able to quickly diagnose the error in the code
"""
from Colours import *

class Error(Exception):
    pass

class PathError(Error):
    """
    Error raised when the path is not a string or the path is empty.
    Furthermore, this error can be raised if the path is not a file or a
    directory
    """
    def __init__(self, mssg):
        self.mssg = red + "ERROR " + reset + mssg


class modeError(Error):
    """
    ERROR raised when the user tries to load  in a mode which is not recognised
    by the programme
    """
    def __init__(self, mssg):
        self.mssg = red + "ERROR " + reset + mssg

