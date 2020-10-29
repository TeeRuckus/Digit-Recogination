import argparse
from Trainer import *
from Image import *
import os
#paths of the located files:
test = '/home/student/test/'
trainning = ''

if __name__ == '__main__':
    parser= argparse.ArgumentParser(description="A tool to claasify house"+
            " numbers given an input images")


