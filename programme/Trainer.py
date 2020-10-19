"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - have the trainer to be able to train against occulisions in the image
"""
from DataSet import *
from abc import abstractmethod

#I have choosen to make this its own object so I can test this class by
#iself, and when the individual trainers inherent from this, I can just
#test the train method for that class
class Trainer(object):
    def __init__(self, **kwargs):
        """
        """
        self._test_set = self.create_data(kwargs['test_path'])
        self._train_set = self.create_data(kwargs['train_path'])
        self._val_set = self.create_data(kwargs['val_path'])

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


    #DATA SEGMENTATION METHODS
    def create_data(self,in_path):
        """
        """
        return Data_Set(in_path)

    @abstractmethod
    def train(self, in_path):
        """
        """
