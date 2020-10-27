"""
AUTHOR: Tawana Kwaramba: 19476700
LAST EDITED:

PURPOSE OF FILE:

TO DO:
    - for tr08 image figure out how to not get the algorithm not to filter out
    the number one
    - filter the images out by color
    -crop image, and get the ROI from the images
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

        bboxes = self.find_clusters(bboxes, 1.10, 0.25)
        bboxes = self.group_clusters(bboxes)

        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "groups of bounding boxes found")

        bboxes = self.filter_areas(bboxes)

        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "filtering by the area")

        #bboxes = self.filter_heights(bboxes)

        #THIS WAS WORKING BEFORE
        bboxes = self.non_max_suppression(bboxes)
        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "non max suppression")

        #filtering the bounding boxes by color, they're just a couple of
        #noise boxes left. Hence, I should be able to filter them out by the
        #colors
        #bboxes = self.filter_colors(bboxes)

        if self._DEBUG:
            self.show_debug_boxes(bboxes, im, "second grouping found")

        left_pt = self.find_leftmost_pt(bboxes)
        right_pt = self.find_leftmost_pt(bboxes, True)

#        new_region = np.array([left_pt[0], left_pt[1], right_pt[0]+right_pt[2],
#            right_pt[1]+right_pt[3]], dtype='int32')

        new_region = self.make_new_region(left_pt, right_pt)

        if self._DEBUG:
            self.show_debug_boxes([new_region], im, "new region found")

    def filter_colors(self, bboxes):
        """
        calculating the average color in each bounding box, and then comparing
        each bounding boxes color with one another, and if a bounding box is
        doesn't follow the average color then that bounding box is considered
        as noise to the image
        """


    def filter_areas(self, bboxes):
        all_areas = []
        #getting all the areas of each bounding box in the image, so I can
        #fileter out the boxes which are most likely going to be outliers
        for box in bboxes:
            all_areas.append(self.find_area(box))

        bboxes = self.remove_outliers(all_areas, bboxes)

        return bboxes

    def get_five_num_summary(self, area_ls):
        min_area, max_area = min(area_ls), max(area_ls)
        percentiles = np.percentile(area_ls, [25, 50, 75])
        #average = np.mean(area_ls)
        summary = [min_area, percentiles[0], percentiles[1], percentiles[2],
                max_area]
        #return min_area, max_area, median, average
        return summary

    def remove_outliers(self, area_ls, bboxes):
        """
        an outlier is a point which will lay siginificantly far away from the
        average
        """
        #if they're going to be only two boxes in the bounding box array
        #they is not point in trying to find the outliers, as one of those
        #boxes will be filtered out which is not what we want
        print(green+"length of box:"+reset, len(bboxes))
        if len(bboxes) > 3:
            #ordering the areas from the smallest area to the largest area
            #area_ls = sorted(area_ls, key=lambda a: a[0])
            #area_ls.sort()
            num_summary = self.get_five_num_summary(area_ls)
            IQR = self.find_IQR(num_summary)
            median = num_summary[2]
            #PLAY AROUND WITH THIS THRESH HOLD
            #the numbers are most likely going to be a similiar size to the
            #median value
            #thresh = 1.05
            #thresh = 1.15
            thresh = 1.20
            #thresh = 1.20
            #area can't be a negative number hence,
            lower_bound = abs(int(median - (thresh * IQR)))
            upper_bound = int(median + (thresh * IQR))

            for area in area_ls:
                #YOU MIGHT HAVE TO BE CAREFUL ABOUT INDX GOING OUT OF RANGE: PLEASE
                #DON'T FORGET TO PUT A GUARD HERE
                indx = area_ls.index(area)

                if area < lower_bound:
                    #I am going to set the index of the outlier indexes to -1 because
                    #if I delete the index straight away it will be hard to keep a
                    #track on which index belongs to which index in the lists
                    bboxes[indx] = [-1, -1, -1, -1]

                if area > upper_bound:
                    #same reasoning applied above applies here aswell
                    bboxes[indx] = [-1, -1, -1, -1]

            bboxes = self.remove_invalid(bboxes)

        return  bboxes


    def find_IQR(self, num_summary):
        """
        finding the inter-quartile range of the obtained data, so we can
        filter out the outliers in the data
        """

        #IQR is the difference between the lower  25 percent quartle, and the
        #upper 75 percent quartile
        IQR = num_summary[3] - num_summary[1]
        return IQR




        for indx, area in enumerate(bboxes):
            pass

    def find_area(self, box):
        """
        regardless of the poisiton of the box the area is going to be always
        w * h
        """
        return box[2] * box[3]

    def non_max_suppression(self, bboxes):
        #sorting boxes by the smallest area to the largest area
        #bboxes = sorted(bboxes, key=lambda area:  bboxes[2] * bboxes[3])
        bboxes = sorted(bboxes, key=lambda x: self.find_area(x))

        ##searching for boxes which are inside one another
        for curr_box_indx, curr_box in enumerate(bboxes):
            x,y,w,h = curr_box
            #ensuring that the algorithm is not comparing to itself, and it's
            #searching relative to the other boxes
            #search_space = bboxes[curr_box_indx:]
            for alt_box in bboxes:
            #for alt_box in search_space:
                #safe-gaurd for trying to delete an index which doesn't
                #exist in the image
                if curr_box_indx < len(bboxes):
                    #is the current box inside any of the alternate boxes
                    x_alt,y_alt,w_alt,h_alt = alt_box
                    end_point_curr_box = x + w
                    end_point_alt_box  = x_alt + w_alt
                    #is the end point of the current box inside the alternate
                    #box and is the beggining x point inside the alternate box
                    #aswell

                    #THIS SHIT IS NEW
#                    diff_end_points = (end_point_curr_box -
#                            end_point_alt_box)
                    #diff_end_points = (end_point_alt_box - end_point_curr_box)

                    if x > x_alt and end_point_curr_box < end_point_alt_box:
                    #if diff_end_points > padding:
                        #is the height of the current box inside the alternate
                        #box
                        height_curr_box = y + h
                        height_alt_box = y_alt + h_alt
#                        height_curr_box_end = end_point_curr_box + h
#                        height_alt_box_end = end_point_alt_box + h_alt

                        #THIS SHIT IS NEW
#                        diff_height_upper = (height_curr_box - height_alt_box)
#                        diff_height_lower = (height_curr_box_end -
#                                height_alt_box_end)
#                        diff_height_upper = (height_alt_box - height_curr_box)
#                        diff_height_lower = (height_alt_box_end -
#                                height_curr_box_end)

                        if height_curr_box < height_alt_box and \
                                y > y_alt:
#                        if height_curr_box < height_alt_box and \
#                        height_curr_box_end > height_alt_box:
#                        if diff_height_upper > padding and \
#                        diff_height_lower > padding:
                            #current box is insde another box, we don't need
                            #this box so we can delete it
                            del bboxes[curr_box_indx]

        return bboxes



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

    def  find_clusters(self, bboxes, thresh_x, thresh_y):
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

                    line_TOL_x  = line_seg_x * thresh_x
                    line_TOL_y = line_seg_y * thresh_y

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
            print(green+"ratio"+reset, ratio)
            if ratio < 1.0 or ratio > 3.5:
            #if ratio <= 1.5 or ratio >= 2.0:
            #if ratio < 1.0 and ratio > 2.0:
                bboxes[indx]  = -1

        return bboxes

    def draw_boxes(self,bboxes, im, color):
        for box in bboxes:
            if box[0] == -1:
                pass
            else:
                x,y,w,h = box
                cv.rectangle(im, (x,y), (x+w, y+h), color, 1)

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
