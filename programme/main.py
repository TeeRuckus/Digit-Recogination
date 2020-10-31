"""
FILENAME: main.py

AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: this is the main of the programme hence, it is to facilate
the functionaility of this assigment. Therefore, it brings all classes created
in order to detect numbers given an input image, crop the digits, save the
necessary files, and to classfity the images
"""
import argparse
from Trainer import *
from Image import *
import os
from ImageLoader import *
from Colours import *
#paths of the located files:
test_path = '/home/student/test/'
trainning_path = '/home/student/train/'

if __name__ == '__main__':
    #extracting the region of interest to the output file
    trainning_images = Image_Loader(trainning_path, 'BGR')
    test_images = Image_Loader(test_path, 'BGR')
    trainner = Trainer(train_path=trainning_path, val_path=test_path,
            mode='BGR')

    results = []
    dists = []
    digits_ls = []
    #im_id is needed so that we can save the files with a unique id but with
    #the same starting string
    for im_id, image in enumerate(test_images):
        try:
            digits = Image(image, im_id)
        #this  is bad programming practice. Although, they is far too many
        #things which can go wrong in terms with the assertions and exceptions
        #thrown by openCV, and the bounding boxes. Therefore, for efficient use
        #of time, I am going to catch all of them
        except:
            #if that image throws an exception, it means that the bounding boxes
            #of that image can't be found and it has failed the extraction
            print(red+"image couldn't extraxt images as"+
                    " bounding boxes couldn't be found"+reset)


        #indexing 1 as I only the result of the classifify as classify returns
        #the result and the distance of the image
        result, other = trainner.classify(digits.im[1])
        base_file_name = 'output/House'
        #creating the file name based on the numbers found
        #I need to use list operations in able to convert the number found in
        #numpy into a whole string
        house_num = result.tolist()
        #the results of the numbers are stored in a 2-dimensional array, whereby
        #they's only  one number inside in the second dimension of the array
        #hence, I am just extracting all those numbers from that second
        #dimension
        house_num = [int(house_num[ii][0]) for ii in range(len(house_num))]
        #taking a house number from an array of strings, and converting it to
        #just a string by itself
        house_num = ''.join(map(str, house_num))
        print(green+"HOUSE NUMBER:"+reset, house_num)
        complete_file_name = base_file_name + house_num + ".txt"

        with open(complete_file_name, 'w') as inStrm:
            inStrm.write('Building {}'. format(house_num))
