from glob import glob
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_closing
from scipy.stats import wasserstein_distance
from skimage.feature import hog
from skimage import data, exposure
from datetime import datetime
import pprint
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
import weakref

from PillClassifierBackend import PillClassifierBackend

if __name__ == "__main__":

    print("Test Code")

    backend = PillClassifierBackend()

    basepath = os.path.dirname(os.path.abspath(__file__))
    resourcespath = '{}/test-patches'.format(basepath)

    patchOne = cv2.imread("{}/Patch1.png".format(resourcespath))
    patchTwo = cv2.imread("{}/Patch2.png".format(resourcespath))
    patchThree = cv2.imread("{}/Patch3.png".format(resourcespath))
    patchFour = cv2.imread("{}/Patch4.png".format(resourcespath))
    patchFive = cv2.imread("{}/Patch5.png".format(resourcespath))
    patchSix = cv2.imread("{}/Patch6.png".format(resourcespath))

    # Patch Area Information

    patches = np.array([patchOne, patchTwo, patchThree,
                        patchFour, patchFive, patchSix])

    patchAreas = np.array([
        backend.GetSizeInformation(patch) for patch in patches
    ])

    print("Patch Area Information----------------------------------")

    for index, patchArea in enumerate(patchAreas):
        print("Patch {} Area: {}".format(index, patchArea))

    # Patch Color Information

    patchHistograms = np.array([
        backend.GetRGBHistogram(patch) for patch in patches
    ])

    print("Patch Color Information----------------------------------")
    print("Histograms were printed")

    '''
    ID = int(input("Enter Patch ID(0 to 4): "))

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histogramPlot = patchHistograms[ID][i]
        plt.plot(histogramPlot, color=col)
        plt.xlim([0, 256])
    plt.show()
    '''

    # Patch Shape Information

    print("Patch Shape Information----------------------------------")

    patchShapes = np.array([
        backend.GetShapeInformation(patch) for patch in patches
    ])

    for index, patchShape in enumerate(patchShapes):
        print("Patch {} Shape: {}".format(index, patchShape))

    print("Vector Space Information----------------------------------")

    featureVectors = []

    for area, shape, histogram in zip(patchAreas, patchShapes, patchHistograms):
        featureVectors.append(
            {
                "shape": shape,
                "area": area,
                "histogram": histogram
            }
        )

    amount = len(featureVectors)
    differenceArray = []
    for pair in itertools.product([i for i in range(amount)], repeat=2):
        pairiter = iter(pair)
        index1 = next(pairiter)
        index2 = next(pairiter)

        histogram1 = featureVectors[index1]["histogram"]
        histogram2 = featureVectors[index2]["histogram"]
        shape1 = featureVectors[index1]["shape"]
        shape2 = featureVectors[index2]["shape"]
        area1 = featureVectors[index1]["area"]
        area2 = featureVectors[index2]["area"]

        currentdistanceHistogram = backend.CompareHistograms(
            histogram1, histogram2)
        currentdistanceShape = shape1 - shape2
        currentdistanceArea = area1 - area2

        differenceArray.append({
            "Shape IDs": "Patch {} and Patch {}".format(index1, index2),
            "Color Difference": currentdistanceHistogram,
            "Shape Difference": currentdistanceShape,
            "Area Difference": currentdistanceArea
        })

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(differenceArray)

    '''
    cv2.imshow('image', patchSix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
