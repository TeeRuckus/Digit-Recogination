import cv2 as cv

def log(item=''):
    #print('CALLED %s' % item)
    print('CALLED {}'.format(item))

def check_types(*args):
    for ii, value in enumerate(args):
        print('%s Type of: % s' % (ii, type(value)))

def check_types_ls(in_ls):
    for ii, value in enumerate(in_ls):
        print('%s Type of: % s' % (ii, type(value)))

def check_sizes(in_ls):
    for ii, value in enumerate(in_ls):
        if len(value.shape) == 3:
            print('%s l: %s w: %s z: %s' % (ii, value.shape[0], value.shape[1], value.shape[2]))
        elif len(value.shape) == 2:
            print('%s l: %s w: %s' % (ii, value.shape[0], value.shape[1]))
        elif len(value.shape) == 1:
            print('%s: l: %s' % (ii, value.shape[0]))
        else:
            print("CAN'T GET SIZE OF IMAGE")

def display_ls(in_ls):
    for ii, thing in enumerate(in_ls):
        print('{}: item: {}'.format(ii, thing))


def display_image(im):
    cv.imshow('display the image', im)
    cv.waitKey()
    cv.destroyAllWindows()
