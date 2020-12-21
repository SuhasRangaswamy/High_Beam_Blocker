import json
import cv2
import numpy as np

def loadDataset(inputPath, truthPath):
    with open(truthPath) as fp:
        truthJson = json.load(fp)

    imageList = truthJson["imageList"]
    listLen = len(imageList)
    truthList = truthJson["dataset"]

    images = []
    masks = []
    for i, image in enumerate(imageList):
        im = cv2.imread(inputPath + "/" + image)
        mask = truthList[i]

        images.append(np.array(im))
        masks.append(np.array(mask))

    return images, masks

def splitDataset(images, masks, percentage=0.8):
    total = len(images)
    splitIndex = int(percentage * total)

    train_images = images[:splitIndex]
    test_images = images[splitIndex:]

    train_masks = masks[:splitIndex]
    test_masks = masks[splitIndex:]

    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    test_images = np.array(test_images)
    test_masks = np.array(test_masks)

    return train_images, train_masks, test_images, test_masks
