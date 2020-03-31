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
import time

from PillClassifierBackend import PillClassifierBackend


def show_image(image_object):

    cv2.imshow('image', image_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return None


if __name__ == "__main__":

    start_time = time.time()

    print("--- Test Branch ---")

    backend = PillClassifierBackend()

    basepath = os.path.dirname(os.path.abspath(__file__))
    resourcespath = '{}/resources'.format(basepath)
    real_image = cv2.imread("{}/real_image.jpg".format(resourcespath))

    objects, object_amount = backend.ExtractObjects(real_image)

    print("{} objects found".format(object_amount))

    # show_image(objects[0])

    print("--- %s seconds ---" % (time.time() - start_time))

    new_object_list_1 = np.array(
        [objects[0], objects[1], objects[2], objects[7], objects[9], objects[12]])

    new_object_list_2 = np.array(
        [objects[3], objects[6]])

    new_object_list_3 = np.array(
        [objects[4], objects[5], objects[8], objects[10], objects[11], objects[13]])

    new_object_list_4 = np.array([objects[0], objects[3], objects[4]])

    '''
    for item in new_object_list_2:
        show_image(item)
    '''
    feature_space = backend.IsAllSame(new_object_list_4)

    if feature_space is None:
        print("Objects are NOT same!")
    else:
        print(feature_space)

    print("--- %s seconds ---" % (time.time() - start_time))
