"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
"""
import unittest
from DataSet import *

class test_DataSet(unittest.TestCase):
    """
    PURPOSE:
    """

    digits_set = Data_Set('../Digits-2020S2/')

    def test_accessors(self):
        #they should be 10 different sets representing each number plus the
        #extra test directory made to tet the mutators
        num_sets = 11
        set_path = '../Digits-2020S2/'
        self.assertEqual(dict, type(self.digits_set.set), "testing if set"+
                " is an actual set")
        self.assertEqual(str, type(self.digits_set.set_path), "testing if"+
                " set path is a list")
        self.assertEqual(set_path, self.digits_set.set_path, "testing if"+
                " the set path is pointing to the right directory")
        self.assertEqual(num_sets, self.digits_set.no_set, "testing if"+
                " data set will return the correct number of sets")
        #the mode hasn't be changed so it should still be loaded in the
        #defualt mode
        self.assertEqual("HSV", self.digits_set.mode, "mode accessors")


    def test_mutators(self):
        nw_set =  '../Digits-2020S2/test_dir/'
        self.digits_set.set = nw_set
        self.assertEqual(nw_set, self.digits_set.set_path, "creating a new"+
                " set")
        #the new set should have a length of three
        self.assertEqual(3, self.digits_set.no_set, "making sure the number"+
                " of sets have been updated appropriately")

    def test_accessing_image_set(self):
        curr_set = self.digits_set.set['1']

        for ii, im in enumerate(curr_set.data):
            cv.imshow("img %s" % ii, im)
        cv.waitKey()
