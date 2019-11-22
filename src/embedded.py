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


class User():
    userAmount = 0

    def __init__(self, username, useruniqueid, isadmin, pillperiods):
        self.username = username
        self.useruniqueid = useruniqueid
        self.isadmin = isadmin
        self.pillperiods = pillperiods
        User.userAmount += 1

    def GetDictionary(self):
        return {
            "Username": self.username,
            "UserUniqueID": self.useruniqueid,
            "IsAdmin": self.isadmin,
            "PillPeriod": self.pillperiods
        }


class PillClass():
    pillClassAmount = 0

    def __init__(self, classname, objectfilename, objectamount):
        self.classname = classname
        self.filename = classname
        self.objectfilename = objectfilename
        self.objectamount = objectamount
        self.objectimage = None
        PillClass.pillClassAmount += 1

    def GetDictionary(self):
        return {
            "ObjectClassName": self.classname,
            "ObjectFileName": self.filename,
            "ObjectAmount": self.objectamount
        }


class Period():

    def __init__(self):
        pass

    def GetDictionary(self):
        return {
            "classSample": 0,
            "lastTake": "0"
        }


class Database():

    def __init__(self, localdatabasefile, onlinedatabaseconfigfile):
        self.localdatabasefile = localdatabasefile
        self.onlinedatabaseconfigfile = onlinedatabaseconfigfile
        self.content = None
        self.firebase = None

    def Initialize(self):
        try:
            self.GetDatabaseContent()
            return True
        except:
            return False

    def GetDatabaseContent(self):
        localdatabasecontent = self.GetLocalDatabase()
        onlinedatabasecontent = dict(self.GetOnlineDatabase())

        if localdatabasecontent["UpdateTime"] == onlinedatabasecontent["UpdateTime"]:
            self.content = localdatabasecontent
            self.statusparameters = self.content['StatusParameters']
            self.users = self.content['Users']
            self.classes = self.content['Classes']
            return True
        else:
            return False

    def GetLocalDatabase(self):
        try:
            with open(self.localdatabasefile, 'r') as file:
                return json.loads(file.read())
        except:
            return False

    def GetOnlineDatabase(self):
        try:
            with open(self.onlinedatabaseconfigfile, 'r') as file:
                filecontent = json.loads(file.read())
                firebasekernel = pyrebase.initialize_app(filecontent)
                self.firebase = firebasekernel.database()

            return self.firebase.get().val()
        except:
            return False

    def GenerateUpdateTime(self):
        now = datetime.now()
        currenttime = now.strftime("%c")

        return currenttime

    def SetDatabaseContent(self):
        try:
            self.content["UpdateTime"] = self.GenerateUpdateTime()
            self.SetLocalDatabase()
            self.SetOnlineDatabase()
            print("Database: Database is saved!")
            return True
        except:
            print("Database: Error occured while saving database!")
            return False

    def SetOnlineDatabase(self):
        try:
            self.firebase.set(self.content)
            return True
        except:
            print("Database: Error occured while updating online database!")

    def SetLocalDatabase(self):
        try:
            json.dump(self.content, codecs.open(self.localdatabasefile, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4)
            return True
        except:
            print("Database: Error occured while updating local database!")

    def CreateNewUser(self):
        pass

    def CreateNewClass(self, pillclassobject):
        self.content["Classes"].append(pillclassobject.GetDictionary())
        self.SetDatabaseContent()
        pass

    def GetClasses(self):
        pass

    def GetUsers(self):
        pass

    def SetUsers(self):
        pass

    def GetStatusParameters(self):
        pass

    def SetStatusParameters(self):
        pass


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
        return cv2.resize(image, None, fx=PillClassifierBackend.resizefactor, fy=PillClassifierBackend.resizefactor, interpolation=cv2.INTER_AREA)

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


class PillClassifier():

    def __init__(self):
        self.backend = PillClassifierBackend()
        self.objectspath = None
        self.localdatabasefile = None
        self.onlinedatabaseconfigfile = None
        self.newshot = None
        self.database = None
        self.users = []
        self.classes = []
        self.pillclasses = []
        self.tempobjectlist = []
        self.temphistogramlist = []
        self.treshold = 15000.0

    def ConnectDatabase(self):
        try:
            self.database = Database(
                self.localdatabasefile,
                self.onlinedatabaseconfigfile
            )

            response = self.database.Initialize()
            if response:
                print("Classifier: Database connection is successful!")
            else:
                print("Classifier: Error occurred while connection database!!")
        except:
            print("Classifier: Error occurred while connection database!")
            print(sys.exc_info()[0])
            raise

    def CheckStatus(self):
        pass

    def TakeaShot(self):
        try:

            # Raspberry Pi Take a Short Function Here!

            newimage = cv2.imread(self.newshot)
            objects, objectamount = self.backend.ExtractObjects(newimage)
            self.tempobjectlist = objects

            print("Classifier: Captured object shape: {}".format(
                self.tempobjectlist.shape))
            return True

        except Exception as e:
            print("Classifier: Error occured while taking new image!")
            print(e)
            return False

    def PostProcessing(self):
        self.temphistogramlist = np.array(
            [self.backend.GetRGBHistogram(object) for object in self.tempobjectlist])

        newclass = PillClass("NewClass", "NewClass", 1)
        self.database.CreateNewClass(newclass)

        print(self.database.content)
        return None

    def Test(self):
        pass


def main():

    # Environment Definitions
    basepath = os.path.dirname(os.path.abspath(__file__))
    resourcespath = '{}/resources'.format(basepath)
    localdatabasefile = '{}/database.json'.format(resourcespath)
    onlinedatabaseconfigfile = '{}/firebase-config.json'.format(resourcespath)
    objectspath = '{}/objects'.format(basepath)
    newshot = '{}/takenimage.jpg'.format(resourcespath)

    # Object Definitions
    pc = PillClassifier()
    pc.localdatabasefile = localdatabasefile
    pc.onlinedatabaseconfigfile = onlinedatabaseconfigfile
    pc.objectspath = objectspath
    pc.newshot = newshot

    # Object Operation
    pc.ConnectDatabase()
    pc.TakeaShot()
    pc.PostProcessing()


if __name__ == "__main__":
    main()
