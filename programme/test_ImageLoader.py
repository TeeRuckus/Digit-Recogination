"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: this is the test code corresponding to ImageLoader.py.
The aim of this file is to validate the functionality of ImageLoader,
and to ensure that's operating as expected

TO DO:
    - Testing of invalid data for all mutators, accesors, and methods
"""
import unittest
from ImageLoader import *

class test_imageLoader(unittest.TestCase):
    """
    PURPOSE: Testing class to test imageLoader
    """

    #class cosntants which we're going to be using throughout our files
    rel_path = '../Digits-2020S2/0/'
    test= Image_Loader(rel_path)

    def test_len(self):
        """
        PURPOSE: testin if the defined class length function will return
        the correct number files in a specified directory
        """
        self.test.path = '../Digits-2020S2/0/'
        #test 1
        num_files = 13
        self.assertEqual(num_files, len(self.test), "number of 0 image files")

    def test_getters(self):
        """
        PURPOSE: testing all the accessors available in the class
        """
        self.test.path = '../Digits-2020S2/0/'
        num_paths = 13
        self.assertEqual(self.rel_path, self.test.path, "testing path getter")
        self.assertEqual('HSV', self.test.mode, "test mode getter")
        self.assertEqual(list, type(self.test.data), "testing if the object "+
                "will return a list of paths")

        self.assertEqual(num_paths, len(self.test.data), "testing the returned"+
                " paths is equal to the number of image files in folder")



    def test_setters(self):
        """
        PURPOSE: testing all the mutators of the given class
        """
        nw_path = '../Digits-2020S2/9/'
        self.test.path = nw_path
        num_files = 10

        self.assertEqual(nw_path, self.test.path, "testin if the new path has"+
                " been updated")

        #to ensure that it does what it suppose to do checking, if the data
        #has been reloaded
        self.assertEqual(num_files, len(self.test), 'testing if the path has'+
                ' been updated properly')

        self.test.mode = 'GRAY'
        self.assertEqual('GRAY', self.test.mode, 'testing if the mode has been'+
                ' set properly')

    def test_display_images(self):
        """
        PURPOSE: visually testing the algorithm to enusre that we can
        display the images out onto the terminal
        """
        self.test.path = '../Digits-2020S2/0/'
        #note: we're expecting it to display the images found in the 0
        #directory, as unittest will destroy any other objects made by
        #previous unit tests, and use the original object which we made
        for ii, img in enumerate(self.test):
            cv.imshow('%s' % ii, img)

        cv.waitKey()

    def test_create_labels(self):
        """
        PURPOSE: to test if this funtion will create the correct labels from the
        given file directories
        """
        #they're going to be 10 labels plus the test directory made
        num_labels = 10
        self.test.path = '../Digits-2020S2/'
        labels = self.test.create_labels()
        print("visual inspection:\n \t %s" % labels)
        self.assertEqual(num_labels,len(labels), "testing the number of created"+
                " labels")


