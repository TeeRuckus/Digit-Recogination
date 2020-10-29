"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - have the trainer to be able to train against occulisions in the image
"""
from ImageLoader import *
from Colours import *
import numpy as np
import pickle

#I have choosen to make this its own object so I can test this class by
#iself, and when the individual trainers inherent from this, I can just
#test the train method for that class
class Trainer(object):
    def __init__(self, **kwargs):
        """
        """
        self._train_path = kwargs['train_path']
        #I am going to use the validation path as the same as the
        #test path for this data
        self._val_path = kwargs['val_path']
        self._trainner = self.train()

    #ACCESORS
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


    #DATA SEGMENTATION METHODS
    def create_data(self,in_path):
        """
        """
        return Data_Set(in_path)

    #@abstractmethod
    def train(self):
        """
        this algorihtm is adapted from:
            Pysource. 2018. "knn handwrittend digits recoginition - OpenCV 3.4
            with python 4 Tutorial 36. https://www.youtube.com/watch?v=tOVwVvRy
            _Pg&ab_channel=Pysource
        """
        #checking if a trainnin file already exists in the current directory
        #if it does, load that file
        trainning_file_name = "kNN_classfier"
        labels_file_name = "kNN_labels"

        if trainning_file_name in os.listdir() and \
        labels_file_name in os.listdir()  :
            print(green+"reading in serilised file...."+reset)
            #load the file
            with open(trainning_file_name, 'rb') as inStrm:
                trainning_data = pickle.load(inStrm)

            with open(labels_file_name, 'rb') as inStrm:
                labels_data = pickle.load(inStrm)
        else:
            print(green+"creating a new file...."+reset)
            #create a new file
            in_path = self._train_path
            trainning_im = Image_Loader(in_path)
            labels = trainning_im.create_labels()

            trainning_data = []
            labels_data =[]

            for label in labels:
                #accessing everything at that given label
                trainning_im.path = in_path +str(label)
                for im in trainning_im:
                    trainning_data.append(im.flatten())
                    labels_data.append(label)
            #knn only accpets numpy arrays which are float32
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

    def classify(self, k=8):
        """
        this algorihtm is adapted from:
            Pysource. 2018. "knn handwrittend digits recoginition - OpenCV 3.4
            with python 4 Tutorial 36. https://www.youtube.com/watch?v=tOVwVvRy
            _Pg&ab_channel=Pysource
        """
        val_path = self._val_path
        test_im = Image_Loader(val_path)

        test_data = []
        for im in test_im:
            test_data.append(im.flatten())
        #knn classifier only accpets numpy arrays
        test_data = np.array(test_data, dype=np.float32)

        ret, result, neigbours, dist = self.trainner.findNearest(test_data, k)
        return result, dist



    #AUGMENTATION OPERATION METHODS
    def rotate_im(self):
        """
        """

    def scale_im(self):
        """
        """

    def bias_im(self):
        """
        """

    def gain_im(self):
        """
        """

    def equalize_hists(self):
        """
        """
