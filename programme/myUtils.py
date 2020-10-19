import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
from debug import *
from DataSet import *
import math
import os
import timeit
np.set_printoptions(threshold=np.inf)


TOL = 0.00001

def clean():
    valid = False
    while not valid:
        proceed = input('Press [N]ext to go onto the next experiment: ')
        if proceed.upper().strip() == 'N':
            valid = True
            cv.destroyAllWindows()
            plt.close('all')

def map_colors(number):
    return {
            0: '#ee7b06',
            1: '#d3d3d3',
            2: '#2f4f4f',
            3: '#556b2f',
            4: '#7f0000',
            5: '#7f0000',
            6: '#008000',
            7: '#d2691e',
            8: '#00008b',
            9: '#daa520',
            10: '#8fbc8f',
            11: '#8b008b',
            12: '#b03060',
            13:	'#ff4500',
            14: '#00ced1',
            15: '#ffff00',
            16:	'#00ff00',
            17: '#8a2be2',
            18: '#00ff7f',
            19: '#e9967a',
            20: '#dc143c',
            21: '#00bfff',
            22: '#0000ff',
            23: '#adff2f',
            24: '#ff00ff',
            25: '#1e90ff',
            26: '#f0e68c',
            27: '#dda0dd',
            28: '#90ee90',
            29: '#ff1493',
            30: '#7b68ee'}[number]

def map_hex2RBG(hex_num):
    """
    CODE ADAPTED FROM: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    """
    hex_num = hex_num.lstrip('#')
    return tuple(int(hex_num[ii:ii+2], 16) for ii in (0,2,4))

def show_img_ls(img_ls, fileName):
    for ii, mat in enumerate(img_ls):
        cv.imshow('image: %s' % ii, mat)
        cv.imwrite(fileName + 'image_%s.jpg' % ii,mat)

def calc_histograms(img, channel=0, hist_size=256, hist_range=(0,256)):
    return cv.calcHist([img], [channel], None, [hist_size], hist_range)

def calc_hist_dist(primary_hist, in_ls, **kwargs):
    if len(kwargs) == 0:
        method=2
        distance = [cv.compareHist(primary_hist,ii,method) for ii in in_ls]
    elif 'method' not in kwin_lis:
        raise KeyError('key word is not supported')
    else:
        distance = [cv.compareHist(primary_hist,ii,kwargs['method']) for ii in in_lis]
    return distance

def show_histograms(image):
    """
    adapted from: https://medium.com/@rndayala/image-histograms-in-opencv-40ee5969a3b7
    """
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])

    plt.show()

def resize_img(img, reduction=0.5):
    img_copy = img.copy()
    #chnage everything to the copy image
    width = img_copy.shape[1] * reduction
    height = img_copy.shape[0] * reduction
    return cv.resize(img_copy, (int(width), int(height)))

def resize_img_dim(img, nw_width, nw_len):
    img_copy = img.copy()
    #chnage everything to the copy image
    return cv.resize(img_copy, (int(nw_width), int(nw_len)))


#TO DO: when you're rotating your images, it's cutting of some of the image
#hence, you need to figure out how to rotate the image and not cut out any
#of the card or the image
def rotate_image(img, angle=45):
    rows, cols, channels = img.shape
    rotated_mat = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0),
            angle,1)
    return cv.warpAffine(img,rotated_mat,(cols,rows))

def rotate_image_b(img, angle=45):
    '''
    METHOD NAME:
    IMPORTS:
    EXPORTS"

    PURPOSE:

    adopted from: #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''
    #grabbing the centre of the image, so we have a point to rotate the image
    (height, width) = img.shape[:2]
    (height_centre, width_centre) = (height // 2, width// 2)
    #negative angle is to get the clockwise rotation of the image
    rotated_mat = cv.getRotationMatrix2D((width_centre, height_centre), -angle,
            1.0)
    #getting the transformed sin, and cosine components of the rotated matrix
    cos = np.abs(rotated_mat[0,0])
    sin = np.abs(rotated_mat[0,1])

    #getting the new bounding dimensions of the image
    new_bounding_height = int((height * cos) + (width * sin))
    new_bounding_width = int((height * sin) + (width * cos))

    rotated_mat[0,2]  += (new_bounding_width / 2) - width_centre
    rotated_mat[1,2] += (new_bounding_height / 2) - height_centre

    #performing the actial rotation on the image, and returning the image
    return cv.warpAffine(img, rotated_mat, (new_bounding_width,
        new_bounding_height))

def SIFT(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    #img = cv.drawKeypoints(img, kp, img,
    #        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = cv.drawKeypoints(img, kp, img)
    return des, kp, img
#SIFT(cv.imread('imgs/diamond2.png'))

def harris(img, thresh, color=[0,0,255]):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    #0.04 - 0.06
    detected_img  = cv.cornerHarris(gray_img, 2, 3, 0.04)
    #you want the raw key points, which hasn't been manipulated in any way
    kp_mat = detected_img
    detected_img = cv.dilate(detected_img, None)
    #filltering the corners we detected by our choosen threshold
    #img[detected_img > 0.01 * detected_img.max()] = color
    img[detected_img > thresh * detected_img.max()] = color
    return img, kp_mat


def count_pixels(img_ls):
    """
    IMPORT:
    EXPORT:
    PURPOSE: to count the how many non-zero pixels exist in an image
    """
    return [np.count_nonzero(ii > 0) for ii in img_ls]

def get_diff_pixels(base_comp, comp_ls):
    return [abs(float(base_comp) - float(ii)) for ii in comp_ls]

def generate_labels(num_images):
    ret = ['experiment image: %s' % ii for ii in range(num_images)]
    ret.insert(0, 'orginal')
    return ret

def save_stats(fileName, area_label_ls, num_labels):
    headers = ['number of labels', 'label', 'Area of label (units^2)']

    with open(fileName, 'w') as inStrm:
        csv_writer = csv.writer(inStrm)
#        csv_writer.writerow([headers[0]])
#        csv_writer.writerow(str(num_labels))
        csv_writer.writerow([headers[0]] + [num_labels])
        csv_writer.writerow(headers[1:])

        for ii, area in enumerate(area_label_ls):
            csv_writer.writerow([ii] + [area])

def save_comparisons(labels, raw_pixels, diff_frm_og, fileName):
    headers = ['Image Name', 'Number of key points', 'difference between orginal keypoints and experiment']
    all_data = zip(labels, raw_pixels, diff_frm_og)
    with open(fileName, 'w') as inStrm:
        csv_writer = csv.writer(inStrm)
        csv_writer.writerow(headers)

        for ii in all_data:
            csv_writer.writerow(ii)

def open_file(fileName):
    os.system('xdg-open %s' % fileName)

def show_diff_hists(base_hist, op_base_hist, op_hists, xLim, **kwargs):
    #showing all the rotated histograms

    ret = plt.figure(kwargs['name'])
    plt.plot(base_hist, color=map_colors(1), label='original image')
    plt.plot(op_base_hist, color=map_colors(2), label='harris orignal image')

    for ii, hist in enumerate(op_hists):
        #need to offset color by 2 as the first two colors were used by the first
        #two images
        color = ii + 2
        plt.plot(hist, color=map_colors(color), label='op: %s' % ii, linestyle='--')

    plt.xlim([0,xLim])
    plt.legend(loc='upper center')
    plt.ylabel('Frequency')
    plt.xlabel('intensity value')
    plt.title(kwargs['name'])
    #plt.show()

    return ret

def show_diff_dist(distance, **kwargs):
    #getting the distances of the rotated image relative to the orginal image
    ret = plt.figure(kwargs['title'])
    labels = ['img: %s' % ii for ii in range(len(distance))]
    labels = tuple(labels)
    y_pos = np.arange(len(labels))
    #distances = [0, 20, 30]

    plt.bar(y_pos, distance, align='center', alpha=0.25)
    plt.xticks(y_pos, labels)
    plt.ylabel('Distance from orginal Harris image')
    plt.xlabel('Distances (units)')
    plt.title(kwargs['title'])

    #plt.show()
    return ret

def crop_img(img, pt1, pt2):
    x_l = int(pt1[0])
    y_l = int(pt1[1])
    x_r = int(pt2[0])
    y_r = int(pt2[1])
    return img[y_l:y_r, x_l:x_r]


def select_key_point(im, **kwargs):
    pass

def pad_image(im, row_pad, col_pad):
    npad = ((row_pad, col_pad), (row_pad, col_pad), (0,0))
    return np.pad(im, pad_width=npad, mode='constant', constant_values=0)


def hog_preprocessing(im):
    """
    IMPORT:
    EXPORT:
    PURPOSE: To rescale the image, so it will be easier, to do the later steps in
    the HOG descriptor steps like the 8x8 box, and the 16x16 box thing later in
    the tutorial
    """
    len_im = im.shape[0]
    width_im = im.shape[1]

    ratio = width_im / len_im
    if abs(ratio - 0.5)  > TOL:
        len_im = width_im * 2
        im = resize_img_dim(im, width_im, len_im)
        nw_ratio  = im.shape[1] / im.shape[0]
        #sanity check to make sure that the image, has scaled to the right
        assert abs(nw_ratio - 0.5) < TOL, 'image is not 1:2 ratio'
    return im

def hog_descriptor(im, **kwargs):
    """
    IMPORT:im (image matrice) a dictonary of all the  hog parameters
    (it doesn't matter which order
    you import them)
    EXPORT: HOG descriptor object

    ASSERTS: returns a hog descriptor object with your given imported parameters
    """

    win_size = im.shape[:2]
    print(win_size)
    #using the convention to set the block size which is typically going to be
    # 2 x cell size
    return cv.HOGDescriptor(win_size,
            kwargs['block_size'], kwargs['block_stride'],
            kwargs['cell_size'], kwargs['num_bins'],
            kwargs['deriv_aperature'], kwargs['win_sigma'],
            kwargs['hist_norm_type'], kwargs['mag_thresh'],
            kwargs['gamma'], kwargs['num_lvls'],
            kwargs['signed_grad'])

"""
TO DO:
    - refactor the diamonds rotated, and the scaled ones, so you can pass
    in whatever image, and you can do the exact same experiement with the
    dugong, and you can just change a couple of parameters
    -you need to add an extra parameter, so you can either choose where to save
    the file too, either the playing card or the dugong
    - do the harris corner detection with the dugong aswell
    - for your sift experiments, you can get the number og key-points they're
    by just getting the length of the list the key-points returned too, and
    comparing if each transform got the same number of keypoints
    -For task 3, when you print out each of the components to the terminal, also number which object is which number
"""

def activity_one_harris_rotated(im, channel, **kwargs):
    """
    IMPORTS:
        - im (String): the name of the imag
        - channel (integer): the channel were you wan to calculate your histograms on
        - kwargs (directionary mapping): importing settings for the harris corner detector
        regardless of the order of the impor. These settings inlcude the following
            - threshold
            - color of the harris corner detector corners in the resultant image

    EXPORTS: None
    PURPOSE: to perform the experiemnts relative to task one of the machine
    percpetion assignment one. Hence, to see if the harris corner detection is
    actually rotation invariant, and to see if the SIFT is scale and rotation
    invariant
    """
    print('-'*20 + 'PERFORMING ROTATIONAL HARRIS EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()

    #the path where all the results of this experiment is kept
    path = 'results/Task_1/rotated_experiements/Harris/%s/' % kwargs['name']

    #creating a list of rotated images with angles of 15 degrees between each image
    rotated_s = [rotate_image_b(img_copy, angle) for angle in range(15,360,15)]

    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    og_harris = harris(img.copy(),kwargs['thresh'], kwargs['color'])[0]

    #creating a list of images which contains the rotate iamges with the harris
    #corner detection performed on each image
    harris_s_rotated = [harris(ii, kwargs['thresh'], kwargs['color'])[0] for ii in rotated_s]

    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_rotated = [calc_histograms(ii,channel, bin_size) for ii in harris_s_rotated]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(),kwargs['thresh'], kwargs['color'])[1]
    harris_s_rotated_kp = [harris(ii, kwargs['thresh'], kwargs['color'])[1] for ii in rotated_s]

    num_kp_og_rotated = count_pixels([og_harris_kp])
    num_kp_rotated = count_pixels(harris_s_rotated_kp)

    #---------------------------------------------------------------------------
    #Experiments for rotated  images
    #---------------------------------------------------------------------------
    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins
    fileName = path + 'comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_rotated[0],num_kp_rotated)
    labels = generate_labels(len(num_kp_rotated))
    save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='Experiment One: Difference between key points: %s' % kwargs['name'])

    open_file(fileName)
    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    exp_two = show_diff_hists(og_hist, og_hist_harris, hists_s_rotated, bin_size,
            name='Experiment two: comparing histograms: %s' % kwargs['name'])

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_rotated)
    exp_three = show_diff_dist(distance, title='Experiment three: Differences between histograms %s'
            % kwargs['name'])

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    harris_s_rotated.insert(0, og_harris)
    show_img_ls(harris_s_rotated, path)

    #showing all the produced plots at once
    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_one_harris_scaled(im, channel, **kwargs):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    print('-'*20 + 'PERFORMING SCALING HARRIS EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    path = 'results/Task_1/scaled_experiements/Harris/%s/' % kwargs['name']
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()
    green = [0,255,0]
    #performing the harris corner detection on the original image, so we have
    #a base point for comparisions latter onwards
    og_harris = harris(img.copy(),kwargs['thresh'], kwargs['color'])[0]

    #create a list of scaled images with each image with a factos difference of 0.0416
    #between each image
    scaled_s = [resize_img(img_copy, factor/24) for factor in range(12, 36, 1)]

    bin_size = 16
    #the histogram of the very orginal image is needed, to confirm that the
    #harris corner detection which introduce variance in the produced histograms
    og_hist = calc_histograms(img.copy(), channel, bin_size)
    og_hist_harris = calc_histograms(og_harris, channel, bin_size)
    hists_s_scaled = [calc_histograms(ii,channel, bin_size) for ii in scaled_s]

    #setting up the appropriate lists to do the comparisons for the keypoints
    #found in each matrix
    og_harris_kp = harris(img.copy(),kwargs['thresh'],kwargs['color'])[1]
    harris_s_scaled_kp = [harris(ii,kwargs['thresh'], kwargs['color'])[1] for ii in scaled_s]

    num_kp_og_scaled = count_pixels([og_harris_kp])
    num_kp_scaled = count_pixels(harris_s_scaled_kp)

    #---------------------------------------------------------------------------
    #Experiments for scaled  images
    #---------------------------------------------------------------------------

    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins
    #fileName = 'results/Task_1/scaled_experiements/Harris/playing_card/comparison.csv'
    fileName = path + 'comparison.csv'
    diff_frm_og = get_diff_pixels(num_kp_og_scaled[0],num_kp_scaled)
    labels = generate_labels(len(num_kp_scaled))
    save_comparisons(labels,  num_kp_scaled, diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='Difference between key points')
    open_file(fileName)

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #change in the count of green pixels found in the image
    exp_two = show_diff_hists(og_hist, og_hist_harris, hists_s_scaled, bin_size,
            name='Experiment two: comparing histograms: %s' % kwargs['name'])


    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist, hists_s_scaled)
    exp_three = show_diff_dist(distance, title='Diffetences between histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_s.insert(0, og_harris)
    show_img_ls(scaled_s, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_one_SIFT_rotated(im, **kwargs):
    """
    IMPORT:
    EXPORT:

    Purpose:
    """
    print('-'*20 + 'PERFORMING ROTATED SIFT EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    path = 'results/Task_1/rotated_experiements/SIFT/%s/' % kwargs['name']
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()

    #so we can calculate the intensities of a black and white image
    channel = 0
    bin_size = 16
    #creating a list of rotated images with angles of 15 degrees between each image
    rotated = [rotate_image_b(img_copy, angle) for angle in range(15,360,15)]

    og_SIFT = SIFT(img_copy)
    rotated_SIFT_Des = [SIFT(ii)[0] for ii in rotated]
    rotated_SIFT_KP = [SIFT(ii)[1] for ii in rotated]
    rotated_SIFT_imgs = [SIFT(ii)[2] for ii in  rotated]

    og_hist_des = calc_histograms(og_SIFT[0], channel, bin_size)
    rotated_hist_des = [calc_histograms(ii, channel, bin_size) for ii in rotated_SIFT_Des]
    #---------------------------------------------------------------------------
    #Experiments for rotated  images
    #---------------------------------------------------------------------------
    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins

    #EXPERIMENT ONE: testing the number of key features extracted
    #finding the number of keypoints found, since keypoints are classes, were're
    #just going to check the length of keypoints returned by each list
    kp_len_og = len(og_SIFT[1])
    kp_lens = [len(ii) for ii in rotated_SIFT_KP]
    diff_frm_og = get_diff_pixels(kp_len_og, kp_lens)
    labels = generate_labels(len(kp_lens))
    #you need to re-factor this so it works for here
    #save_comparisons(labels, num_kp_rotated,diff_frm_og, fileName)
    exp_one = show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')
    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    exp_two = show_diff_hists(0, og_hist_des, rotated_hist_des, bin_size,
            name='Showing the difference between the obtained descriptors')

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, rotated_hist_des)
    exp_three = show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    rotated_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(rotated_SIFT_imgs, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()

    cv.waitKey()
    clean()

def activity_one_SIFT_scaled(im, **kwargs):
    """
    IMPORTS:
    EXPORTS:
    PURPOSE:
    """
    print('-'*20 + 'PERFORMING SCALING SIFT EXPERIEMENTS: %s' % kwargs['name'] + '-'*20)
    img = cv.imread(im)
    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------
    #creating copies, as you want to do calculations and manipulations based on
    #copies. Just as an extra  precautious step
    img_copy = img.copy()
    og_SIFT = SIFT(img_copy)

    path = 'results/Task_1/scaled_experiements/SIFT/%s/' % kwargs['name']
    channel = 0
    bin_size = 16

    scaled = [resize_img(img_copy, factor/24) for factor in range(12, 36, 1)]

    scaled_SIFT_Des = [SIFT(ii)[0] for ii in scaled]
    scaled_kp = [SIFT(ii)[1] for ii in scaled]
    scaled_SIFT_imgs = [SIFT(ii)[2] for ii in scaled]

    og_hist_des = calc_histograms(og_SIFT[0], channel, bin_size)
    scaled_des_hists = [calc_histograms(ii, channel, bin_size) for ii in scaled_SIFT_Des]

    #---------------------------------------------------------------------------
    #Experiments for scaled  images
    #---------------------------------------------------------------------------

    #EXPERIMENT ONE: checking if the harris corner detection picked up the same
    #poins

    kp_len_og = len(og_SIFT[1])
    kp_lens = [len(ii) for ii in scaled_kp]
    diff_frm_og = get_diff_pixels(kp_len_og, kp_lens)
    labels = generate_labels(len(kp_lens))
    exp_one = show_diff_dist(diff_frm_og, title='the difference of keypoints found relative to first image')

    #EXPERIMENT TWO: plotting the histograms of the image, to see if they is a
    #to see how the intensities of the descriptors are changing throughout the experiment
    #placing a zero in the first parameter, as we don't have a base image, as
    #we're going to be caclulating the histograms of the descriptors
    exp_two = show_diff_hists(0, og_hist_des, scaled_des_hists, bin_size,
            name='Showing the difference on the obtained descriptors')

    #EXPERIMENT THREE: calculating the distances between the produced histograms
    #taking advantage of the image, the only thing green on the image is the
    #detected points
    distance = calc_hist_dist(og_hist_des, scaled_des_hists)
    exp_three = show_diff_dist(distance, title='Differences between the histograms')

    #EXPERIMENT FOUR: a visual inspection to ensure that the same points
    #were found across the generated images relative to the first image
    #produced
    scaled_SIFT_imgs.insert(0, og_SIFT[2])
    show_img_ls(scaled_SIFT_imgs, path)

    exp_one.show()
    exp_two.show()
    exp_three.show()
    cv.waitKey()
    clean()

def activity_two_SIFT_rotated(im, pt1, pt2, **kwargs):
    im = cv.imread(im)
    im_copy = im.copy()

    feature  = crop_img(im_copy, pt1, pt2)
    cv.imshow('original image', im_copy)
    rotated_features = [rotate_image(feature, ii) for ii in range(15,360,15)]
    og_sift  = SIFT(im_copy)[2]
    sift_des = [SIFT(ii)[2] for ii in rotated_features]

    print(og_sift.shape)
    check_sizes(sift_des)

    sift_des.insert(0, og_sift)
    #comp = [cv.norm(og_sift - ii) for ii in sift_des]

    show_img_ls(sift_des,"results/Task_2/%s/rotation/" %kwargs['name'])
    cv.waitKey()
    clean()

def activity_two_SIFT_scaled(im, pt1, pt2, **kwargs):
    im = cv.imread(im)
    im_copy = im.copy()

    scaled_features = [resize_img(im_copy, factor/24 ) for factor in range(12,36,1)]
    cropped_features = [crop_img(ii, pt1, pt2) for ii in scaled_features]

    feature = crop_img(im, pt1, pt2)

    og_sift = SIFT(im_copy)[2]
    sift_des = [SIFT(ii)[2] for ii in cropped_features]

    sift_des.insert(0, og_sift)
    show_img_ls(sift_des, "results/Task_2/%s/scaling/" %kwargs['name'])
    #comp = [cv.norm(og_sift - ii) for ii in sift_des]
    #exp_one = show_diff_dist(comp, title='the difference between sift scaled descrptors')

    cv.waitKey()
    clean()

def activity_two_hog_scaled(im, pt1, pt2):
    im = cv.imread(im)
    im_copy = im.copy()

    scaled_features = [resize_img(im_copy, factor/24) for factor in range(12,36,1)]
    cropped_features = [crop_img(ii, pt1, pt2) for ii in scaled_features]

    feature = crop_img(im_copy, pt1, pt2)
    cv.imshow('feature', feature)

    hog = cv.HOGDescriptor()
    des_og = hog.compute(feature)

    h_ls = [hog.compute(ii) for ii in cropped_features[7:]]
    comp = [cv.norm(des_og - ii) for ii in h_ls]

    exp_one = show_diff_dist(comp, title='Showing the difference between orignal descriptors and scaled descriptors')

    exp_one.show()
    cv.waitKey()
    clean()

def activity_two_hog_rotated(im, pt1, pt2):
    im = cv.imread(im)
    im_copy = im.copy()

    cv.imshow('original', im_copy)

    #-------------------------------------------------------------------------------
    #SET UP
    #-------------------------------------------------------------------------------

    #I have choosen the two as the intresting keypoint hence, extracting the two
    #pre-processing: the image must have a ratio of 1:2 for the hog to work
    #properly

    #need to pad the image, so we can extract the two by itself without the
    #diamond, and to maintain a ratio of 1:2
    feature = crop_img(im_copy, pt1, pt2)
    print(feature.shape)
    cv.imshow('feature', feature)

    rotated_features = [rotate_image(feature, ii) for ii in range(15,360,15)]

    hog = cv.HOGDescriptor()
    des_og = hog.compute(feature.copy())
    h_ls = [hog.compute(ii) for ii in rotated_features]
    comp = [cv.norm(des_og - ii) for ii in h_ls]
    exp_one = show_diff_dist(comp, title='showing the difference between orignal hog descriptor and rotated hog descriptor')

    exp_one.show()
    cv.waitKey()
    clean()


def display_kp_ls(in_ls):
    for ii, num_kp in  enumerate(in_ls):
        print('image: {}, {} key points found'.format(ii,num_kp))


def activity_three(im, invert_threshold=False, **kwargs):
    """
    Adapted from: #https://iq.opengenus.org/connected-component-labeling/#:~:text=Connected%20Component%20Labeling%20can%20be,connectedComponents()%20function%20in%20OpenCV.&text=The%20function%20is%20defined%20so,path%20to%20the%20original%20image.
    """
    name = kwargs['im_name']
    path ='results/Task_3/%s/' % name.lower()
    imgs = []
    #task i
    im = cv.imread(im)
    im_copy = im.copy()
    gray_img = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    #PRE-PROCESSING
    cv.imshow('original gray image', gray_img)
    #blurring the image to remove any potential noise
    blur = cv.GaussianBlur(gray_img, (5,5),0)
    cv.imshow('blurred image', blur)
    thresh = cv.threshold(blur, 0, 255,  cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    if invert_threshold:
        thresh = thresh.max() - thresh

    #cv.imshow('threshold', thresh)
    imgs.append(thresh)

    #applyting the connected component labelling algorithm
    connectivity=8
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity,cv.CV_32S)

    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    #set bg label to black
    labeled_img[label_hue==0] = 0

    #drawing the contours onto the image
    contours  = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(im_copy, contours[1], -1, (0,255,0), 3)

    #cv.imshow('after component labelling: %s' % kwargs['im_name'], labeled_img)
    imgs.append(labeled_img)

    #task ii)
    #fileName ='results/Task_3/%s/results_for_%s.csv' % (name.lower(), name)
    fileName = path + 'resutls_for_%s.csv' % name
    area_of_all_labels = [stats[ii][cv.CC_STAT_AREA] for ii in range(num_labels)]
    save_stats(fileName, area_of_all_labels, labels.max())
    open_file(fileName)

    show_img_ls(imgs, path)
    cv.waitKey()
    clean()

    return labels, area_of_all_labels, centroids

def activity_four_contours(im, thresh, **kwargs):
    """
    CODE ADAPTED FROM: https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
    """
    imgs = []
    path = 'results/Task_4/%s/contours/' % (kwargs['name'].lower().strip())
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray =  cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    canny_trans = cv.Canny(im_gray, thresh, thresh * 2)

    contours, hierarchy = cv.findContours(canny_trans, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[1:]

    #drawing the found contours onto the image

    #setting up a black canvas, the size of the image. To draw the picture onto
    drawing_canvas = np.zeros((canny_trans.shape[0], canny_trans.shape[1], 3), dtype=np.uint8)


    for ii in range(len(contours)):
        #randint is exlusive hence, it's actually doing numbers from 0 - 255
        color  = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing_canvas, contours, ii, color, 1, cv.LINE_AA, hierarchy,0)

    imgs.append(drawing_canvas)
    show_img_ls(imgs, path)
    cv.waitKey()
    clean()
    #cv.watershed(im_res, markers)


#apparently the HSV color scheme is better for image detection
def activity_four_kMeans(raw_im, im, **kwargs):
    imgs = []
    path = 'results/Task_4/%s/%s/' % (kwargs['name'].lower().strip(), kwargs['color_space'].upper().strip())

    im_flat = im.reshape((-1,3))
    im_flat = np.float32(im_flat)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, kwargs['num_iter'] , kwargs['eplison'])

    K = kwargs['K']
    attempts = kwargs['num_iter']

    #flags  = cv.KMEANS_RANDOM_CENTERS
    flags = cv.KMEANS_PP_CENTERS
    labels, (centers) = cv.kmeans(im_flat, K, None, criteria, attempts, flags)[1:]

    centers = np.uint8(centers)

    labels = labels.flatten()

    #putting together the segmented image
    seg_im = centers[labels.flatten()]

    #converting it back to the original image shape
    seg_im = seg_im.reshape(im.shape)


    #cv.imshow('the segmented image', seg_im)
    imgs.append(seg_im)

    #showing each of the segments individually

    #for ii in range(labels.max()):
    for ii in  range(3):
        #making a copy, so the mask is only applied once onto the image
        im_mask = raw_im.copy()
        im_mask = im_mask.reshape((-1,3))
        #setting any segment in the image which corresponds to that cluster to blue
        im_mask[labels == ii] = [255,0,0]
        #converting the flattened pixle matrices into the original image
        im_mask = im_mask.reshape(im.shape)

        #cv.imshow('cluster %s: %s' % (ii, kwargs['color_space']), im_mask)
        imgs.append(im_mask)

    #a quick way for me to save the images
    show_img_ls(imgs, path)
    cv.waitKey()
    clean()

#I AM GOING TO COME BACK TO THIS, IT'S CAUSING ME A HEADACHE
def activity_four_watershed(im, invert_threshold=False):
    """
    CODE ADAPTED FROM: https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    """
    im = cv.imread(im)
    im_copy = im.copy()
    im_gray = cv.cvtColor(im_copy, cv.COLOR_BGR2GRAY)

    #Part one - Preprocessing the removal of noise from the image
    blur = cv.GaussianBlur(im_gray, (5,5), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    if invert_threshold:
        thresh = thresh.max() - thresh

    cv.imshow('threshold image %s' % invert_threshold, thresh)

    #noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh , cv.MORPH_OPEN, kernel, iterations = 2)

    #sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    #finding sure foreground area
    dist = cv.distanceTransform(opening, cv.DIST_L2,5)
    sure_fg = cv.threshold(dist, 0.7 * dist.max(), 255,0)[1]

    #finding the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    #marker labelling
    markers  = cv.connectedComponents(sure_fg)[1]

    #adding 1 to labels to ensure that background is not 0 but 1
    markers += 1

    #marking the unkown region with zero
    markers[unknown==255] = 0

    markers = cv.watershed(im_copy, markers)
    im_copy[markers == - 1] = [255,0,0]

    cv.imshow('resultant', im_copy)
    cv.waitKey()
    clean()

def check_image_size():
    all_data = Data_Set('../Digits-2020S2/')

    all_labels = all_data.set.keys()
    print(all_labels)


    for set_num in all_labels:
        print("set %s" % set_num)
        for ii, im in enumerate(all_data.set[set_num].data):
            print("\t id {} {}".format(ii, im.shape))


if __name__ == '__main__':
    check_image_size()
