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
    PATCH_SIZE = 100

    def __init__(self):
        print("Pill Classifier Backend, Version: {}".format(
            __class__.VERSION))

    @staticmethod
    def show_image(image):

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    @staticmethod
    def is_all_same(object_list):
        result = True
        histograms = np.array(
            [__class__.get_rgb_color(item) for item in object_list])

        sizes = np.array(
            [__class__.get_size_info(item) for item in object_list])

        shapes = np.array(
            [__class__.get_shape_info(item) for item in object_list])

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
    def is_all_similar(feature_space):

        max_distance = 0

        for pair in itertools.combinations(feature_space, r=2):

            distance = cdist(pair[0].reshape(-1, 1).transpose(),
                             pair[1].reshape(-1, 1).transpose(), 'euclidean')[0][0]

            if (distance > __class__.THRESHOLD):
                return (False, distance)

            if (distance > max_distance):
                max_distance = distance

        return (True, max_distance)

    @staticmethod
    def get_resized_image(image):
        return cv2.resize(
            image,
            None,
            fx=__class__.RESIZE_FACTOR,
            fy=__class__.RESIZE_FACTOR,
            interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def mask_object(image, mask):

        mask = cv2.copyMakeBorder(
            mask,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])

        image = cv2.copyMakeBorder(
            image,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            __class__.PATCH_SIZE,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])

        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        _, binary_mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV)
        binary_mask = 255 - binary_mask
        binary_mask = np.expand_dims(binary_mask, axis=-1)
        masked_image = cv2.bitwise_and(
            image, image, mask=np.uint8(binary_mask))

        return masked_image[cY-__class__.PATCH_SIZE:cY+__class__.PATCH_SIZE, cX-__class__.PATCH_SIZE:cX+__class__.PATCH_SIZE]

    @staticmethod
    def extract_objects(image):
        resized_image = __class__.get_resized_image(image)

        # Step: 1 - Find edges
        grayscaled_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gaussian_image = cv2.GaussianBlur(grayscaled_image, (7, 7), 0)
        unsharped_image = cv2.addWeighted(
            grayscaled_image, 2, gaussian_image, -1, 0)
        edged_image = cv2.Canny(unsharped_image, 70, 450)

        # Step: 2 - Connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edged_image = cv2.dilate(edged_image, kernel, iterations=1)
        edged_image = cv2.erode(edged_image, kernel, iterations=1)

        # Step: 2 - Fill closed loops
        edged_image = 255 - edged_image
        _, thresholded_image = cv2.threshold(
            edged_image, 250, 255, cv2.THRESH_BINARY_INV)
        floodfill = thresholded_image.copy()
        height, width = thresholded_image.shape[:2]
        mask = np.zeros((height+2, width+2), np.uint8)
        cv2.floodFill(floodfill, mask, (0, 0), 255)
        inverted_image = cv2.bitwise_not(floodfill)
        mask = thresholded_image | inverted_image

        # Step: 3
        _, markers = cv2.connectedComponents(mask)
        object_amount = np.max(markers)
        temp_mask = np.array([])
        (width, height) = markers.shape

        for marker in range(1, object_amount + 1):

            def check_pixel(pixel):
                if (pixel == marker).any():
                    return 255
                else:
                    return 0

            temp_list = np.array(
                list(map(check_pixel, markers.flatten())))

            temp_list = np.uint8(temp_list)
            temp_mask = np.append(temp_mask, temp_list)

        temp_mask = temp_mask.reshape(object_amount, width, height)

        objects = np.array([
            __class__.mask_object(resized_image, mask) for mask in temp_mask
        ])

        return (objects, object_amount)

    @staticmethod
    def get_shape_info(image):

        width, height, _ = image.shape

        offsetX = int(width / 2)
        offsetY = int(height / 2)

        rotated = ndimage.rotate(image, 45)

        width, height, _ = rotated.shape

        centerX = int(width / 2)
        centerY = int(height / 2)

        rotated = rotated[centerX-offsetX:centerX +
                          offsetX, centerY-offsetY: centerY+offsetY]

        sum_of_images = image + rotated
        area_of_original_image = __class__.get_size_info(image)
        area_of_sum_of_images = __class__.get_size_info(sum_of_images)

        result = ((area_of_sum_of_images - area_of_original_image) /
                  area_of_original_image)*100

        return result

    @staticmethod
    def get_size_info(image):

        grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(
            grayscaled_image, 1, 255, cv2.THRESH_BINARY)
        width, height = binary_image.shape

        pixel_count = width * height

        flatten_image = binary_image.flatten()

        nonzero_pixel_count = 0
        for pixel in flatten_image:
            if pixel >= 1:
                nonzero_pixel_count += 1

        return nonzero_pixel_count

    @staticmethod
    def get_rgb_histogram(image):

        rgb_histogram = []

        color = ('b', 'g', 'r')
        for channel, col in enumerate(color):
            histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
            rgb_histogram.append(histogram)

        return np.array(rgb_histogram)

    @staticmethod
    def compare_histograms(first_histogram, second_histogram):
        return cdist(first_histogram.reshape(-1, 1).transpose(),
                     second_histogram.reshape(-1, 1).transpose(), 'cityblock')[0][0]

    @staticmethod
    def get_rgb_color(image):

        grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(
            grayscaled_image, 10, 255, cv2.THRESH_BINARY_INV)

        width, height = thresholded_image.shape

        coefficient = int(
            ((np.count_nonzero(thresholded_image))/(width * height))*100)

        B, G, R, _ = cv2.mean(image)

        mean_color = np.array([R, G, B]) * 92

        return mean_color
