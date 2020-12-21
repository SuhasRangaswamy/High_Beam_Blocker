import cv2
import argparse
import numpy as np
import imutils
from skimage import measure

inputPath = 'Ramurthynagar.MP4'
outPath = './Images_masks'

def extractImages(inPath, outputPath):
    '''
    Function to extract frames from video file
    and save it as .jpg images.
    :param inPath: Path to input video file
    :param outputPath: Path to where to save the images
    :return: None
    '''
    count= 0
    cap = cv2.VideoCapture(inPath)
    success, image = cap.read()
    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, (count*100))
        success, image = cap.read()
        if success and count < 2:
            print('Reading a new frame: ', success)
            print('frame%d' % count)
            resized = imutils.resize(image, width=300)
            ratio = image.shape[0] / float(resized.shape[0])

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # threshold the image to reveal light regions in the blurred image
            thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)[1]

            # perform a series of erosions and dilations to remove
            # any small blobs of noise from the thresholded image
            thresh = cv2.erode(thresh, None, iterations=1)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # perform a connected component analysis on the thresholded
            # image; then initialize a mask to store only the "large"
            # components
            labels = measure.label(thresh, neighbors=8, background=0)
            mask = np.zeros(thresh.shape, dtype="uint8")

            # loop over the unique components
            for label in np.unique(labels):
                # if this is the background label, ignore it
                if label == 0:
                    continue

                # otherwise, construct the label mask and count the
                # number of pixels
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)

                # if the number of pixels in the component is sufficiently
                # large, then add it to our mask of "large blobs"
                if numPixels > 40:
                    mask = cv2.add(mask, labelMask)

            #cv2.imwrite(outputPath + "/frame%d.jpg" % count, mask)
            count += 1

if __name__ == "__main__":
    '''
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="Path to video file")
    a.add_argument("__pathOut", help="Path to where to save images")
    args = a.parse_args()

    extractImages(args.pathIn, args.pathOut)
    '''
    extractImages(inputPath, outPath)