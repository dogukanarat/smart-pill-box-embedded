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

    VERSION = 1.0
    RESIZE_FACTOR = 1
    THRESHOLD = 500

    def __init__(self):
        print("Pill Classifier Backend, Version: {}".format(
            __class__.VERSION))

    @staticmethod
    def show_image(image_object):

        cv2.imshow('image', image_object)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    @staticmethod
    def IsAllSame(objectlist):
        result = True
        histograms = np.array(
            [__class__.GetRGBColor(item) for item in objectlist])

        sizes = np.array(
            [__class__.GetSizeInformation(item) for item in objectlist])

        shapes = np.array(
            [__class__.GetShapeInformation(item) for item in objectlist])

        def feature_mapping(item):
            return [int(item[0][0]), int(item[0][1]), int(item[0][2]), int(item[1]), int(item[2])]

        feature_space = np.array([feature_mapping(item)
                                  for item in zip(histograms, sizes, shapes)])

        for pair in itertools.combinations(feature_space, r=2):

            distance = cdist(pair[0].reshape(-1, 1).transpose(),
                             pair[1].reshape(-1, 1).transpose(), 'euclidean')[0][0]

            if (distance > __class__.THRESHOLD):
                return None

        if result:
            return feature_space

    @staticmethod
    def ResizeImage(image):
        return cv2.resize(
            image,
            None,
            fx=__class__.RESIZE_FACTOR,
            fy=__class__.RESIZE_FACTOR,
            interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def MaskObject(image, mask):

        PATCH_SIZE = 100

        mask = cv2.copyMakeBorder(
            mask, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        image = cv2.copyMakeBorder(
            image, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        _, binarymask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)
        binarymask = 255 - binarymask
        binarymask = np.expand_dims(binarymask, axis=-1)
        maskedimage = cv2.bitwise_and(image, image, mask=np.uint8(binarymask))

        return maskedimage[cY-PATCH_SIZE:cY+PATCH_SIZE, cX-PATCH_SIZE:cX+PATCH_SIZE]

    @staticmethod
    def ExtractObjects(image):
        resizedimage = PillClassifierBackend.ResizeImage(image)

        # Step: 1 - Find edges
        grayscaled = cv2.cvtColor(resizedimage, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(grayscaled, (7, 7), 0)
        unsharped = cv2.addWeighted(grayscaled, 2, gaussian, -1, 0)
        edged = cv2.Canny(unsharped, 70, 450)

        # Step: 2 - Connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        edged = cv2.erode(edged, kernel, iterations=1)

        # Step: 2 - Fill closed loops
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
    def GetShapeInformation(image):

        width, height, _ = image.shape

        offsetX = int(width / 2)
        offsetY = int(height / 2)

        rotated = ndimage.rotate(image, 45)

        width, height, _ = rotated.shape

        centerX = int(width / 2)
        centerY = int(height / 2)

        rotated = rotated[centerX-offsetX:centerX +
                          offsetX, centerY-offsetY: centerY+offsetY]

        sumOfImages = image + rotated
        areaOfOriginalImage = PillClassifierBackend.GetSizeInformation(image)
        areaOfSumOfImages = PillClassifierBackend.GetSizeInformation(
            sumOfImages)

        result = ((areaOfSumOfImages - areaOfOriginalImage) /
                  areaOfOriginalImage)*100

        return result

    @staticmethod
    def GetSizeInformation(image):

        grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binaryImage = cv2.threshold(
            grayscaleImage, 1, 255, cv2.THRESH_BINARY)
        width, height = binaryImage.shape

        pixelcount = width * height

        flattenImage = binaryImage.flatten()

        nonzeroPixelCount = 0
        for pixel in flattenImage:
            if pixel >= 1:
                nonzeroPixelCount += 1

        return nonzeroPixelCount

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

    @staticmethod
    def GetRGBColor(image):
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY_INV)

        width, height = threshed.shape

        coefficient = int(((np.count_nonzero(threshed))/(width * height))*100)

        B, G, R, _ = cv2.mean(image)

        mean_color = np.array([R, G, B]) * 92

        return mean_color
