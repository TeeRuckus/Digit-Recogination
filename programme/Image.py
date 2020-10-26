"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - have find  left most point filter out coordinates with -1 as an index
"""

from abc import abstractmethod
import numpy as np
from Errors import *
from Colours import *
import cv2 as cv

class Image(object):
    def __init__(self, im):
        self._DEBUG = False
        #self._DEBUG = True
        self._im = self.get_ROI(im)

    @property
    def im(self):
        return self._im

    @property
    def DEBUG(self):
        return self._DEBUG

    @im.setter
    def im(self, in_im):
        self._im = self.get_ROI(in_im)

    def debug(self):
        """
        PURPOSE: to act as a toggle to witch the debugging features for this
        class
        """
        if self.DEBUG:
            self._DEBUG = False
        else:
            self._DEBUG = True

    def get_ROI(self, im, **kwargs):
        im = self._validate_image(im)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        #decreasing the required memory the image needs but still keeping the
        #important features
        gray = cv.GaussianBlur(gray, (5,5), 0)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

        edge_thresh = 100
        #the openCV doc recommends that you will have your upper thresh hold
        #twice as the lower threshold
        canny_trans = cv.Canny(thresh, edge_thresh, edge_thresh * 2)
        rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        canny_trans = cv.erode(thresh, None, iterations=1)
        canny_trans = cv.dilate(thresh, None, iterations=1)
        #canny_trans_invert = canny_trans.max() - canny_trans

        if self._DEBUG:
            cv.imshow("found edges after morphology" , canny_trans)
            #cv.imshow("the inversion of that image", canny_trans_invert)
            cv.waitKey()
            cv.destroyAllWindows()

        mser = cv.MSER_create(35)
        regions, bboxes = mser.detectRegions(canny_trans)
        #trying to filter the bounding boxes in relation to the heights, and the width

        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "original bounding boxes found")

        bboxes = self.filter_bounding_boxes(bboxes)
        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "filtered bounding boxes")

        bboxes = self.find_clusters(bboxes)
        bboxes = self.group_clusters(bboxes)
        #bboxes = self.non_max_suppression(bboxes)
        #bboxes = self.filter_heights(bboxes)
        #bboxes = self.filter_areas(bboxes)

        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "groups of bounding boxes found")

        left_pt = self.find_leftmost_pt(bboxes)
        right_pt = self.find_leftmost_pt(bboxes, True)

#        new_region = np.array([left_pt[0], left_pt[1], right_pt[0]+right_pt[2],
#            right_pt[1]+right_pt[3]], dtype='int32')

        print(red+"left point"+reset, left_pt)
        print(red+"right point"+reset, right_pt)
        new_region = self.make_new_region(left_pt, right_pt)

        if self._DEBUG:
            self.show_debug_boxes([new_region], im, "new region found")

        #make new image

    def non_max_suppression(self, bboxes):
        #sorting boxes by the smallest area to the largest area
        bboxes = sorted(bboxes, key=lambda area:  bboxes[2] * bboxes[3])

    def area_bboxes(self, bboxes):
        #regardless where the bounding box is located the area is always
        #going to be w * h
        return bboxes[2] * bboxes[3]

    def make_new_region(self, left_box, right_box):
        print(red+"left_box"+reset, left_box)
        print(red+"left_box"+reset, right_box)
        #top left corner of the left-most upper most point in the image
        #coordinates
        x = left_box[0]
        y = left_box[1]

        #right most lower most corner in the image coordiantes
        x_r =  right_box[0] + right_box[2]
        y_r = right_box[1] + right_box[3]

        w = x_r - x
        h = y_r - y

        return np.array([x, y, w, h], dtype='int32')



    def show_debug_boxes(self, bboxes, im, title):
            debug_im = im.copy()
            blue = (255,0,0)
            self.draw_boxes(bboxes, debug_im, blue)
            cv.imshow(title, debug_im)
            cv.waitKey()
            cv.destroyAllWindows()

    #FUNCTIONS WHICH WILL HELP TO FIND THE INTEREST AREA
    def resize_image(self, im, x, y):
        return cv.resize(im, (int(x), int(y)))

    def group_clusters(self, clusters):
        """
        many boxes where founf by the find_clusters algorithm, this function
        responsibility it to clean up the bounding boxes found hence, to select
        the biggest box out of the cluster
        """
        cluster_b = []

        for indx, cluster in enumerate(clusters):
            box_one = cluster[0]
            box_two = cluster[1]

            x_1, y_1, w_1, h_1 = box_one
            x_2, y_2, w_2, h_2 = box_two

            #getting the longer lines out of the two boxes parsed in
            nw_h = max(h_1, h_2)
            nw_w = max(w_1, w_2)

            #getting the left-most x and y points so we know where the clusters
            #are going to begin
            nw_x = min(x_1, x_2)
            nw_y = min(y_1, y_2)

            #finding the gap between the clustered boxes
            #the gap is given by the difference between the end point of the
            #left most box and the box next to it
            gap_x = abs(x_2 - (x_1 + w_1))
            gap_y = abs((y_2 + h_2) - h_1)

            nw_box = np.array([nw_x, nw_y, nw_w, nw_h], dtype='int32')
            clusters[indx] = nw_box

            #I have finished analysing the search space now, I want ot move onto
            #the next pair. This is to avoid dupilcates, and doubling up
            #of parameters in the programme
        return clusters

    def  find_clusters(self, bboxes):
        """
        the numbers which are in the image should be relatively close to each
        other hence, we're going to get rid of all the boxes which are not
        close to each other because the chance of these boxes not been a number
        is very high
        """
        cluster = []

        bboxes =  sorted(bboxes, key=lambda x: x[0])
        bboxes = self.remove_invalid(bboxes)

        for start, curr_box in enumerate(bboxes):
            x,y,w,h = curr_box
            pt1 = (x, y)
            pt2 = (x+w, y+h)
            search_region = bboxes[start:]
            for alt_box in search_region:
                if len(search_region) > 0:
                    x_alt,y_alt,w_alt,h_alt = alt_box
                    pt1_alt = (x_alt,y_alt)
                    pt2_alt =(x_alt+w_alt, y_alt+h_alt)

                    x_diff = abs(pt2[0] - pt1_alt[0])
                    y_diff = abs(pt2[1] - pt2_alt[1])

#                    line_seg_x = max(pt2[0], pt2_alt[0])
#                    line_seg_y = max(pt2[1], pt2_alt[1])
                    line_seg_x = max(w, w_alt)
                    line_seg_y = max(h, h_alt)

                    line_TOL_x  = line_seg_x * 1.10
                    line_TOL_y = line_seg_y * 0.25

                    #if x_diff < pt2[0]  and x_diff < pt2_alt[0]:
                    #I don't think you're doing this comparison correctly if I
                    #am being honest
                    if x_diff <= line_TOL_x:
                            #if y_diff < h:
                            if y_diff <= line_TOL_y:
                                cluster.append([curr_box, alt_box])
        return cluster

    def filter_bounding_boxes(self, bboxes):
        """
        we know that for the bounding boxes which will contain the digits
        the height is going to be longer than the width
        """
        for indx, box in  enumerate(bboxes):
            x,y,w,h = box
            pt1 = (x, y)
            pt2 = (x+w, y+h)

            #if (abs(pt1[0] - pt2[0]) >= abs(pt1[1] - pt2[1])):
            if w >= h:
                bboxes[indx] = -1

            ratio = h/w
            #we're going to expect the height of the digits to be no more than
            #250% of the widht of the image, and the length to be no less
            #than the width of the bounding box
            if ratio < 1.0 or ratio > 3.5:
                bboxes[indx]  = -1

        return bboxes

    def draw_boxes(self,bboxes, im, color):
        for box in bboxes:
            if box[0] == -1:
                pass
            else:
                x,y,w,h = box
                cv.rectangle(im, (x,y), (x+w, y+h), color, 2)

    def find_leftmost_pt(self, bboxes, reverse=False):
        """
        if reverse is true, it will find the right-most lower most point of the
        bounding boxes

        Notes:
            - if a box is inside another box, it's going to find the box  inside
            because it's comparing in relation to the left point of the box
        """
        bboxes = self.remove_invalid(bboxes)
        left_most_boxes = sorted(bboxes, key=lambda x: x[0], reverse=reverse)

        temp_box = left_most_boxes[0]

        #CASE 1: clear left most box will be met if it fails CASE 2's and
        #CASE 3's checks

        #CASE 2: boxes are the same x-coordinate hence to enusre that
        #the upper-most box is selected
        for box in left_most_boxes:
            #case: when two boxes have the same x-dimension but differing
            #y-dimensions
            if temp_box[0] == box[0]:
                if temp_box[1] > box[1]:
                    temp_box = box

        #CASE 3: the left most box is selected but if they's a box which is
        #higher than the current box combine find the intersecting points
        highest_boxes = sorted(bboxes, key=lambda y: y[1], reverse=reverse)
        highest_box = highest_boxes[0]

        equal = highest_box == temp_box
        #if the current box is not the highest box, form an intersection
        #with the highest box
        if not equal.all():
            temp_box = self.find_intersection(highest_box, temp_box,
                    reverse=reverse)

        if self._DEBUG:
            print('='*80)
            print(red, 'find_leftmost_p() | temp box',reset)
            print('\t {}'.format(temp_box))
            print('='*80)

        return temp_box[0], temp_box[1], temp_box[2], temp_box[3]

    def remove_invalid(self, bboxes):
        if self._DEBUG:
            print('='*80)
            print(red + "og array" + reset, bboxes)
            print('='*80)

        nw_bboxes = []
        for indx, box in enumerate(bboxes):
            #if the first index is equal to -1 the whole box will equal to -1,
            #and that will be an invalid box
            if box[0] == -1:
                pass
            else:
                nw_bboxes.append(box)

        #converting the new list into an array, so openCV function can use
        #this array to draw the boxes
        bboxes = np.array(nw_bboxes, dtype='int32')

        if self._DEBUG:
            print('='*80)
            print(red + "resultant array" +reset, bboxes)
            print('='*80)

        return bboxes

    def find_intersection(self, box_one, box_two, reverse=False):
        """
        IMPORT:
        EXPORT:

        PURPOSE
            if you're trying to find the right most corner, set reverse to
            true, and add the width, and the height of the object of interset
        """
        temp_boxes = [box_one, box_two]
        #placing the box with the lowest x value at the front
        temp_boxes = sorted(temp_boxes, key=lambda x: x[0], reverse=reverse)
        #the first boxes x coordinate
        nw_x = temp_boxes[0][0]
        #the right most point will be the temp box's in reverse, and it
        #will be the that boxes x value plus that boxes w value
        nw_w = temp_boxes[0][2]
        #placing the box withthe lowest y value at the front
        temp_boxes = sorted(temp_boxes, key=lambda  y: y[1], reverse=reverse)
        #the first boxes y coordinate
        nw_y = temp_boxes[0][1]
        #the right most point will be the temp boxes in reverse, and it
        #will be that  box's y value plus box's h value
        nw_h = temp_boxes[0][3]

        if self._DEBUG:
            print('='* 80)
            print(red, 'find_intersection() | interesction',reset)
            print('\t {}, {}, {}, {}'.format(nw_x, nw_y, nw_w, nw_h))
            print('='* 80)

        return nw_x, nw_y, nw_w, nw_h

    def draw_boxes(self, bboxes, im, color=(0,0,255)):
        """
        IMPORT:
        EXPORT:

        PURPOSE: it's to draw a bounding box which is either a shape of a
        rectangle or a square
        """
        for box in bboxes:
            #if the box has been labelled by a negative -1 by a filtering
            #algorithm we should skip over this box
            if box[0] == -1:
                pass
            else:
                x,y,w,h = box
                cv.rectangle(im, (x,y), (x+w, y+h), color, 2)

    def _validate_image(self, in_im):
        #an image is going to be an numpy matrice
        if not type(in_im) == np.ndarray:
            #all loaded images, are an unsigned interger by defualt
            if not in_im.dtype == 'uint8':
                raise ImageError("Error: an image wasn't laoded in the system")

        return in_im
