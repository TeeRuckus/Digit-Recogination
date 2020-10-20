import cv2 as cv
import numpy as np
import os
np.set_printoptions(threshold=np.inf)
def extract_grounds(im):
    bg_colour = 0
    fg_colour = 255
    extracted_labels = []
    rows, cols = im.shape

    for ground in range(1, max_label_no + 1):
        extracted_labels.append([[255 if labels[xx][yy] == label_no else 0 for yy
            in range(cols)] for xx in range(rows)])

    return extracted_labels

def draw_boxes(bboxes, im):
    for box in bboxes:
        if box[0] == -1:
            pass
        else:
            x,y,w,h = box
            print('first coordinate:\n {} {}'.format(x,y))
            print('distnace between them:\n {} {}'.format(w,h))
            cv.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)

all_imgs = os.listdir('../train_updated/')
interest = ["tr17.jpg"]

for ii, path in  enumerate(all_imgs):
    im = cv.imread('../train_updated/' + path)
    im_copy = im.copy()
    im_copy_2 = im.copy()

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)
    #cv.imshow("blurred image %s" %ii, gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    #cv.imshow("threshold image %s" %ii, thresh)

    edge_thresh = 100
    canny_trans = cv.Canny(thresh, edge_thresh, edge_thresh*2 )
    #cv.imshow("canny_trans before %s" %ii, canny_trans)

    rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    canny_trans = cv.erode(thresh, None, iterations=1)
    canny_trans = cv.dilate(thresh, None, iterations=1)
    canny_trans_invert = canny_trans.max() - canny_trans


    #cv.imshow("found edges after morphology %s" %ii, canny_trans)

    mser = cv.MSER_create(35)
    regions, bboxes = mser.detectRegions(canny_trans)
    print(type(bboxes))
    print('before: \n {}'.format(bboxes))

    #trying to filter the bounding boxes in relation to the heights, and the widths
    #we know that for the bounding boxes which will contain the digits
    #the height is going to be longer than the with

    #filetering the bounding boxes whihc are most likely not going to contain
    #digits in them
    for indx, box in  enumerate(bboxes):
        x,y,w,h = box
        pt1 = (x, y)
        pt2 = (x+w, y+h)
        #if (y + h) >= (x + w):
        if (abs(pt1[0] - pt2[0]) >= abs(pt1[1] - pt2[1])):
            bboxes[indx] = -1

    print('after filtering\n ', bboxes)

    for box in bboxes:
        if box[0] == -1:
            pass
        else:
            x,y,w,h = box
            print('first coordinate:\n {} {}'.format(x,y))
            print('distnace between them:\n {} {}'.format(w,h))
            cv.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)

    cv.imshow('bounding boxes filered %s' %ii, im)

    #combing filtered boxes to create area where the numbers are most likely going to be

    #making one big bounding box of the area
    print('boxes before: \n', bboxes)
    bboxes = np.concatenate(bboxes)
    print('big box: \n ', bboxes)
    #draw_boxes(bboxes, im_copy_2)
    #cv.imshow('approximate 1: %s' % ii, im_copy_2)




    #making blank black templates to place the background and foreground
    #features on
#    bg_img = np.zeros_like(canny_trans)
#    fg_img = np.zeros_like(canny_trans)
#    rows, cols = canny_trans.shape


    #do this if you want to invert the image for some reason
    #same = canny_trans.max() - canny_trans


    #cv.imshow('same results %s' % ii, same)
    #
#
#    cnts, heirarchy = cv.findContours(canny_trans, cv.RETR_TREE,
#            cv.CHAIN_APPROX_SIMPLE)[1:]
#
#    copy = im.copy()
#    copy2 = im.copy()
#    cv.drawContours(copy, cnts, -1, (0,255,0), 3)
#    cv.imshow("all contorus %s" %ii, copy)
#
#    cnts = sorted(cnts, key =cv.contourArea, reverse = True )
#    total_cnts = len(cnts)
#    keep = total_cnts * 0.7
#    keep = int(keep)
#
#    #the contours of the actual numbers is not going to be the biggest or the
#    #smallest, so I am going to get rid of half of the contours mainly
#    #focusing on the smaller contour
#    cnts = cnts[:keep]
#    print(len(cnts))
#    cv.drawContours(copy2, cnts, -1, (0,0,255), 2)
#    cv.imshow("image after filtered contours %s" %ii, copy2)


cv.waitKey()
cv.destroyAllWindows()

"""
im = cv.imread('../train_updated/tr01.jpg')
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
rect_kern = cv.getStructuringElement(cv.MORPH_RECT, (13,5))
black_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rect_kern)
cv.imshow('blackhat', black_hat)
cv.imshow('normal image', im)

sq_kern = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
light = cv.morphologyEx(gray, cv.MORPH_CLOSE, sq_kern)
light = cv.threshold(light, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

cv.imshow('light regions', light)

#applying the scharr gradient, to be able to get more emphasises on the numbers
#of the house
gradx = cv.Sobel(black_hat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=1)
gradx = np.absolute(gradx)

(minVal, maxVal) = (np.min(gradx), np.max(gradx))
gradx = 255 * ((gradx - minVal) / (maxVal - minVal))
gradx = gradx.astype("uint8")
cv.imshow("scharr", gradx)

gradx = cv.GaussianBlur(gradx, (5,5), 0)
gradx = cv.morphologyEx(gradx, cv.MORPH_CLOSE, rect_kern)
thresh = cv.threshold(gradx, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

cv.imshow("Grad thresh", thresh)

#clean up the gradient image a little bit
thresh = cv.erode(thresh, None, iterations=2)
thresh = cv.dilate(thresh, None, iterations=2)
cv.imshow('grad erode/dilaet', thresh)

thresh = cv.bitwise_and(thresh, thresh, mask=light)
thresh = cv.dilate(thresh, None, iterations=2)
thresh = cv.erode(thresh, None, iterations=1)

cv.imshow('resultant', thresh)

mser = cv.MSER_create()
regions = mser.detectRegions(gradx)[0]

hulls = [cv.convexHull(p.reshape(-1,1,2)) for p in regions]
cv.polylines(im, hulls, 1, (0,255,0))

cv.imshow('detected regions', im)

cv.waitKey()
vis = cv.GaussianBlur(vis, (7,7), 0)
cv.imshow('blurred image', vis)
edges_canny = cv.Canny(vis, 0, 200)

vis = cv.HoughLinesP(edges_canny, 1, np.pi/180, 50, None, 50, 10)
draw_lines_p(im, vis)

cv.imshow('detected_lines', im)
cv.waitKey()
cv.destroyAllWindows()

cv.imshow('detected_lines', im)
cv.waitKey()
cv.destroyAllWindows()
mser = cv.MSER_create()
regions = mser.detectRegions(vis)[0]

hulls = [cv.convexHull(p.reshape(-1,1,2)) for p in regions]
cv.polylines(im, hulls, 1, (0, 255, 0))

cv.imshow('detected regions', im)
cv.waitKey()
cv.destroyAllWindows()
"""
