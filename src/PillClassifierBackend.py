from glob import glob
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_closing
from scipy.stats import wasserstein_distance
from skimage.feature import hog
from skimage import data, exposure
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random
import uuid
import warnings
import codecs
import json
import pyrebase
import itertools
import asyncio
import csv


class PillClassifierBackend():

    version = 1.0
    resizefactor = 0.2
    treshold = 1500

    def __init__(self):
        print("Pill Classifier Backend, Version: {}".format(
            PillClassifierBackend.version))

    @staticmethod
    def IsAllSame(objectlist):
        amount, _, _, _ = objectlist.shape
        result = True

        for pair in itertools.product([i for i in range(amount)], repeat=2):
            pairiter = iter(pair)
            index1 = next(pairiter)
            index2 = next(pairiter)
            histogram1 = __class__.GetRGBHistogram(objectlist[index1])
            histogram2 = __class__.GetRGBHistogram(objectlist[index2])
            currentdistance = __class__.CompareHistograms(
                histogram1, histogram2)

            if (currentdistance < __class__.treshold):
                result = False

        return result

    @staticmethod
    def ResizeImage(image):
        return cv2.resize(
            image,
            None,
            fx=PillClassifierBackend.resizefactor,
            fy=PillClassifierBackend.resizefactor,
            interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def MaskObject(image, mask):
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        _, binarymask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)
        binarymask = 255 - binarymask
        binarymask = np.expand_dims(binarymask, axis=-1)
        maskedimage = cv2.bitwise_and(image, image, mask=np.uint8(binarymask))

        return maskedimage[cY-100:cY+100, cX-100:cX+100]

    @staticmethod
    def ExtractObjects(image):
        resizedimage = PillClassifierBackend.ResizeImage(image)
        # Step: 1
        grayscaled = cv2.cvtColor(resizedimage, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(grayscaled, (7, 7), 0)
        unsharped = cv2.addWeighted(grayscaled, 2, gaussian, -1, 0)
        edged = cv2.Canny(unsharped, 70, 800)

        # Step: 2
        edged = 255 - edged
        _, threshed = cv2.threshold(
            edged, 250, 255, cv2.THRESH_BINARY_INV)
        floodfill = threshed.copy()
        h, w = threshed.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(floodfill, mask, (0, 0), 255)
        inverted = cv2.bitwise_not(floodfill)
        mask = threshed | inverted

        # Step: 3
        _, markers = cv2.connectedComponents(mask)
        objectamount = np.max(markers)
        tempmask = np.array([])
        (width, height) = markers.shape

        for marker in range(1, objectamount + 1):

            def CheckPixel(pixel):
                if (pixel == marker).any():
                    return 255
                else:
                    return 0

            templist = np.array(
                list(map(CheckPixel, markers.flatten())))

            templist = np.uint8(templist)
            tempmask = np.append(tempmask, templist)

        tempmask = tempmask.reshape(objectamount, width, height)

        objects = np.array([PillClassifierBackend.MaskObject(
            resizedimage, mask) for mask in tempmask])

        return (objects, objectamount)

    @staticmethod
    def GetRGBHistogram(image):

        rgbhistogram = []

        color = ('b', 'g', 'r')
        for channel, col in enumerate(color):
            histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
            rgbhistogram.append(histogram)

        return np.array(rgbhistogram)

    @staticmethod
    def CompareHistograms(firsthistogram, secondhistogram):
        return cdist(firsthistogram.reshape(-1, 1).transpose(),
                     secondhistogram.reshape(-1, 1).transpose(), 'cityblock')[0][0]
