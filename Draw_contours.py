from imutils import contours
import numpy as np
import imutils
import cv2
from shapedetector import ShapeDetector
from threading import Thread
import sys

# import Queue class from Python 3
if sys.version_info >= (3,0):
    from queue import Queue

# otherwise, imoprt the Queue class for Python 2.7
else:
    from Queue import Queue

class DrawContours:
    def __init__(self, queueSize=128):
        # Boolean to indicate if the thread should be stopped or not
        self.stopped = False
        self.ratio = 0
        # initialize the queue used to store frames read from the video file
        self.Q1 = Queue(maxsize=queueSize)
        self.Q2 = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to draw contours on the frames from the video file
        t = Thread(target=self.draw,args=())
        t.daemon = True
        t.start()
        return self

    def saveMask(self, mask):
        # otherwise, ensure the queue has room in it
        if not self.Q1.full():
            # add the mask in queue
            self.Q1.put(mask)

    def saveFrame(self, frame):
        if not self.Q2.full():
            self.Q2.put(frame)
            
                
    def readMask(self):
        # return next mask in the queue
        return self.Q1.get()

    def readFrame(self):
        return self.Q2.get()
                
    def more(self):
        # return true if there are still masks in the queue
        return self.Q1.qsize() > 0 and self.Q2.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def draw(self, ratio):
        self.ratio = ratio
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set , stop the thread
            if self.stopped:
                return

            mask = self.readMask()
            frame = self.readFrame()
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
                c *= self.ratio
                c = c.astype("int")

                #draw the bright spot on the image
                (x,y,w,h) = cv2.boundingRect(c)
                ((cX,cY), radius) = cv2.minEnclosingCircle(c)
                #cv2.drawContours(image,[c],-1,(0,255,0),2)
                #cv2.circle(image,(int(cX), int(cY)), int(radius),(0,0,255),1)
                shape = sd.detect(c)

                
                cv2.drawContours(frame, [c], -1, (0,255,0), 2)
                #cv2.putText(frame, shape, (int(cX),int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # draw the center of the contour
                #cv2.circle(frame,(int(cX),int(cY)),4,(0,0,255),1)
                #cv2.putText(image, "#{}".format(i+1),(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

            # show the output image
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
