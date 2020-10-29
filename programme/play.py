import cv2 as cv
import numpy as np
import  numpy.ma as ma
from Image import Image

some_ids = ['one', 'two', 'three', 'four']
system_value = [1, 2, 3,4]

for ids, values in zip(some_ids, system_value):
    if values  ==  2:
        print("found")
        del values
        del ids


#I need to unzip the list object
print(some_ids)
print(system_value)
