import cv2
import glob
import json
import imutils
import numpy as np
from skimage import measure

class datasetCreator:
    def __init__(self, input_path, output_path):
        self.datasetPath = input_path
        self.outputFilePath = output_path
        self.truthDataset = []
        self.imageList = []
        self.dataset = {}

    def load_images(self, numImagesToLoad):
        imList = glob.glob(self.datasetPath + "/*.jpg")
        im_len = len(imList)
        if numImagesToLoad < im_len:
            print("Number of images specified to load is more then the Total data itself!!!")
            print("Total number of dataset is ", im_len)
        count = 0
        for image in imList:
            if count <= numImagesToLoad:
                im = cv2.imread(image)
                im_name = image.split("/")[-1]
                print(im_name, " ", count)
                self.imageList.append(im_name)

                resized = cv2.resize(im, (512, 512), interpolation=cv2.INTER_NEAREST)

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
                labels = measure.label(thresh, background=0)
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

                self.truthDataset.append(mask.tolist())
                count += 1
        self.dataset = {'dataset': self.truthDataset, 'imageList': self.imageList}

    def exportDataset(self):
        with open(self.outputFilePath + "/truth.json", 'w+') as f:
            json.dump(self.dataset, f)

if __name__=="__main__":
    input = "../datasets/Images"
    outputFile = "../datasets/"
    datasetCtr = datasetCreator(input, outputFile)
    datasetCtr.load_images(100)
    datasetCtr.exportDataset()
