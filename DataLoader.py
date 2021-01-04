import json
import cv2
import numpy as np

def loadDataset(inputPath, truthPath):
    with open(truthPath) as fp:
        truthJson = json.load(fp)

    imageList = truthJson["imageList"]
    listLen = len(imageList)
    truthList = truthJson["dataset"]

    dataset = []
    for i, image in enumerate(imageList):
        im = cv2.imread(inputPath + "/" + image)
        im = cv2.resize(im, (512, 512))
        mask = truthList[i]

        dt = np.dstack((im, mask))

        dataset.append(np.array(dt))
    np.random.shuffle(np.array(dataset))
    return dataset, listLen

def splitDataset(dt, percentage=0.8):
    total = len(dt)
    splitIndex = int(percentage * total)

    train_dt = dt[:splitIndex]
    test_dt = dt[splitIndex:]

    return train_dt, test_dt

def seperateDatasets(dataset):
    images = []
    masks = []
    for data in dataset:
        images.append(data[:,:,0:3])
        masks.append(data[:,:,-1])
    images = np.array(images)
    masks = np.array(masks)
    return images, masks
