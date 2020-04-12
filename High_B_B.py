from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from shapedetector import ShapeDetector
import glob
'''
#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to the image file")
args = vars(ap.parse_args())
'''
d = 1
#file = r'High_beam_pictures\camera_pics\5.jpg'
for file in glob.iglob(r'High_beam_pictures\camera_pics\*.jpg'):
    #load the image, resize and convert it to grayscale, and blur it
    print(file)
    image = cv2.imread(file)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)

    # threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(blurred,230,255,cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh,None, iterations=2)

    # perform a connected component analysis on the thresholded
    # image; then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        #if this is the background label, ignore it
        if label == 0:
            continue

        #otherwise, construct the label mask and count the
        #number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 40:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnt = contours.sort_contours(cnts)[0]

    sd = ShapeDetector()

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # multiply the contour (x,y)-coordinates by the resize ratio,
        #then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")

        #draw the bright spot on the image
        (x,y,w,h) = cv2.boundingRect(c)
        ((cX,cY), radius) = cv2.minEnclosingCircle(c)
        #cv2.drawContours(image,[c],-1,(0,255,0),2)
        #cv2.circle(image,(int(cX), int(cY)), int(radius),(0,0,255),1)
        shape = sd.detect(c)

        
        cv2.drawContours(image, [c], -1, (0,255,0), 2)
        cv2.putText(image, shape, (int(cX),int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        # draw the center of the contour
        cv2.circle(image,(int(cX),int(cY)),4,(0,0,255),1)
        cv2.putText(image, "#{}".format(i+1),(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

          
    # show the output image
    #cv2.imshow("Image", thresh)

    filename = "High_beam_pictures/camera_pics/output_5/image_%d.jpg"%d
    cv2.imwrite(filename,image)
    d = d+1
    cv2.waitKey(0)
