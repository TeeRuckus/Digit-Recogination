"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE: the purpose of the file is to facilate the trainning of the
digits for this programme, and it's also to take in some in-coming digits and
classify those digits in relation to the trainned data. This file utilise
kNN for trainning and classification of its data
"""
from Image import *
from ImageLoader import *
from Colours import *
import numpy as np
import pickle

#This class is meant to be an abstract class but, #I have choosen to make this
#its own object so I can test this class by iself, and when the individual
#trainers inherent from this, I can just test the train method for that specific
#trainning method

class Trainer(object):
    def __init__(self, **kwargs):
        self._mode = kwargs['mode']
        self._train_path = kwargs['train_path']
        #I am going to use the validation path as the same as the
        #test path for this data as they're doing the same thing
        self._val_path = kwargs['val_path']
        self._trainner = self.train()

    #===========================ACCESORS========================================
    @property
    def test_set(self):
        return self._test_set

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def trainner(self):
        return self._trainner


    #===========================PUBLIC METHODS==================================
    def train(self):
        """
        import:None
        Export: knn (kNN clusters objext)

        PURPSOSE: it is to train the kNearest neigbour classsify given the
        trainning data of digits from 0 to 9

        this algorihtm is adapted from:
            Pysource. 2018. "knn handwrittend digits recoginition - OpenCV 3.4
            with python 4 Tutorial 36. https://www.youtube.com/watch?v=tOVwVvRy
            _Pg&ab_channel=Pysource
        """
        trainning_file_name = "kNN_classfier"
        labels_file_name = "kNN_labels"

        #checking if a trainning file already exists in the current directory
        #if it does, load that file

        if trainning_file_name in os.listdir() and \
        labels_file_name in os.listdir()  :
            print(green+"reading in serilised file...."+reset)
            #load the file if it exists, this will allow for faster
            #classification times if the module has been already pre-trainned before
            with open(trainning_file_name, 'rb') as inStrm:
                trainning_data = pickle.load(inStrm)

            with open(labels_file_name, 'rb') as inStrm:
                labels_data = pickle.load(inStrm)
        else:
            print(green+"creating a new file...."+reset)
            #create a new file
            in_path = self._train_path
            #using image loading object for faster and efficient image loading
            trainning_im = Image_Loader(in_path, self._mode)
            #creating the labels based on the file name which the number belongs
            #too
            labels = trainning_im.create_labels()

            trainning_data = []
            labels_data =[]

            #pre-pranning out trainning images, and our trainning data
            for label in labels:
                #opening each digit file and storing it at the trainning_im
                #object
                trainning_im.path = in_path +str(label)
                #accessing every image inside the trainning_im object
                for im in trainning_im:
                    trainning_data.append(im.flatten())
                    labels_data.append(label)
            trainning_data = np.array(trainning_data, dtype=np.float32)
            labels_data = np.array(labels_data, dtype=np.float32)

            with open(trainning_file_name, 'wb') as inStrm:
                #creating a serilised file for future use
                pickle.dump(trainning_data, inStrm)

            with open(labels_file_name, 'wb') as inStrm:
                #writing a labels serilised file for future use
                pickle.dump(labels_data, inStrm)

        knn = cv.ml.KNearest_create()
        knn.train(trainning_data,cv.ml.ROW_SAMPLE, labels_data)

        return knn

    def classify(self, images, k=8):
        """
        IMPORT: images (list of uint8 numpy arrays i.e. images)
        EXPORT: results (string): the label  which the classify to the images
                dist (numpy array): contains the L2 norm distance of each
                result found

        PURPOSE: it's to assign a label to inputted images

        this algorihtm is adapted from:
            Pysource. 2018. "knn handwrittend digits recoginition - OpenCV 3.4
            with python 4 Tutorial 36. https://www.youtube.com/watch?v=tOVwVvRy
            _Pg&ab_channel=Pysource
        """
        test_data = []
        for im in images:
            #this needs to be the same size as the provided trainning data
            im = Image.resize_image(self,im, 28, 40)
            test_data.append(im.flatten())
        #knn classifier only accpets numpy arrays
        test_data = np.array(test_data, dtype=np.float32)

        ret, result, neigbours, dist = self.trainner.findNearest(test_data, k)
        return result, dist



    #AUGMENTATION OPERATION METHODS
    def add_noise(self, im):
        """
        IMPORT: im (numpy array data type: uint8)
        EXPORT: im (numpy array data type: uint8)

        PURPOSE: it's to add noise to an image, to stop the trainner to
        over-fitting to the provided trainning data
        """
