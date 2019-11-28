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
import weakref

from PillClassifierBackend import PillClassifierBackend


class User():
    userAmount = 0
    instances = []

    def __init__(self, username, useruniqueid, isadmin, pillperiods=[]):
        self.username = username
        self.useruniqueid = useruniqueid
        self.isadmin = isadmin
        self.pillperiods = pillperiods
        User.userAmount += 1
        self.__class__.instances.append(weakref.proxy(self))

    def GetDictionary(self):
        return {
            "Username": self.username,
            "UserUniqueID": self.useruniqueid,
            "IsAdmin": self.isadmin,
            "PillPeriod": self.pillperiods
        }


class PillClass():
    pillClassAmount = 0
    instances = []

    def __init__(self, classname, samplepath, amount, uniqueclassname=None):
        self.classname = classname
        self.samplepath = samplepath
        self.amount = amount
        self.objectimage = None
        self.RGBHistogram = None
        self.GetRGBHistogramExecuted = False
        self.uniqueclassname = uniqueclassname

        if self.uniqueclassname == None:
            self.uniqueclassname = str(uuid.uuid4())

        PillClass.pillClassAmount += 1
        self.__class__.instances.append(weakref.proxy(self))

    def GetDictionary(self):
        return {
            "ClassName": self.classname,
            "Amount": self.amount,
            "UniqueClassName": self.uniqueclassname
        }

    def GetRGBHistogram(self):
        if not self.GetRGBHistogramExecuted:
            histogram = PillClassifierBackend.GetRGBHistogram(
                self.GetSampleImage()
            )
            self.RGBHistogram = histogram
            self.GetRGBHistogramExecuted = True
            return self.RGBHistogram
        else:
            return self.RGBHistogram

    def GetSampleImage(self):
        try:
            if self.objectimage == None:

                sampleimage = cv2.imread(
                    "{}/{}.png".format(self.samplepath, self.uniqueclassname)
                )
                self.objectimage = sampleimage

            else:
                None

            return self.objectimage

        except Exception as e:
            print(e)
            return False

    def StoreSampleImage(self, sampleimage):
        try:
            cv2.imwrite(
                "{}/{}.png".format(self.samplepath, self.uniqueclassname),
                sampleimage
            )
            print("New Sample Stored: {}/{}.png".format(self.samplepath,
                                                        self.uniqueclassname))

        except Exception as e:
            print(e)
            return False


class PillPeriod():
    periods = []
    instances = []

    def __init__(self, useruniqueid, classname, lasttake):
        self.relateduser = relateduser
        self.classname
        self.lasttake = lasttake
        self.__class__.instances.append(weakref.proxy(self))

    def GetDictionary(self):
        return {
            "ClassName": self.classname,
            "LastTake": self.lasttake
        }


class Database():

    def __init__(self, localdatabasefile, onlinedatabaseconfigfile, objectspath):
        self.localdatabasefile = localdatabasefile
        self.onlinedatabaseconfigfile = onlinedatabaseconfigfile
        self.objectspath = objectspath
        self.content = None
        self.firebase = None
        self.users = []
        self.pillclasses = []

    def Initialize(self):
        try:
            self.GetDatabaseContent()

            self.statusparameters = self.content['StatusParameters']

            if bool(self.content['Classes']):
                for pillclass in self.content['Classes']:
                    pillclassobject = PillClass(
                        pillclass["ClassName"],
                        self.objectspath,
                        pillclass["Amount"],
                        uniqueclassname=pillclass["UniqueClassName"])
                    self.pillclasses.append(pillclassobject)

            if bool(self.content['Users']):
                for user in self.content['Users']:

                    pillperiods = []

                    if bool(user['PillPeriods']):
                        for pillperiod in user['PillPeriods']:
                            pillperiodobject = PillPeriod(
                                user["UserUniqueID"],
                                pillperiod["ClassName"],
                                pillperiod["LastTake"]
                            )
                            pillperiods.append(pillperiodobject)

                    userobject = User(
                        user["Username"],
                        user["UserUniqueID"],
                        user["IsAdmin"],
                        pillperiods
                    )

                    self.users.append(userobject)

            return True
        except Exception as e:
            print(e)
            return False

    def GetDatabaseContent(self):
        localdatabasecontent = self.GetLocalDatabase()
        onlinedatabasecontent = dict(self.GetOnlineDatabase())

        if localdatabasecontent["UpdateTime"] == onlinedatabasecontent["UpdateTime"]:
            self.content = localdatabasecontent
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
        self.pillclasses.append(pillclassobject)
        self.content["Classes"].append(pillclassobject.GetDictionary())
        self.content["StatusParameters"]
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


class PillClassifier():

    def __init__(self):
        self.treshold = 15000.0
        self.backend = PillClassifierBackend()
        self.objectspath = None
        self.localdatabasefile = None
        self.onlinedatabaseconfigfile = None
        self.newshot = None
        self.database = None
        self.users = []
        self.pillclasses = []
        self.tempobjectlist = []
        self.temphistogramlist = []

    def ConnectDatabase(self):
        try:
            self.database = Database(
                self.localdatabasefile,
                self.onlinedatabaseconfigfile,
                self.objectspath
            )

            response = self.database.Initialize()
            if response:
                self.pillclasses = self.database.pillclasses
                self.users = self.database.users
                print("Classifier: Database connection is successful!")
            else:
                print("Classifier: Error occurred while connection database!")
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

        except:
            print("Classifier: Error occured while taking new image!")
            print(sys.exc_info()[0])
            return False

    def PostProcessing(self):
        self.temphistogramlist = np.array([
            self.backend.GetRGBHistogram(object) for object in self.tempobjectlist
        ])

        localhistogramlist = np.array([
            pillclassobject.GetRGBHistogram() for pillclassobject in self.pillclasses
        ])

        distances = []

        for temphistogramlist in self.temphistogramlist:
            for localhistogram in localhistogramlist:
                distances.append(self.backend.CompareHistograms(
                    temphistogramlist,
                    localhistogram
                ))

        if(min(distances) < self.treshold):
            print("Classifier: Given pill class is found in the database!")

        '''
        for index, tempobject in enumerate(self.tempobjectlist):

            newpillclass = PillClass("NewClass{}".format(
                index), self.objectspath, 1)

            newpillclass.StoreSampleImage(tempobject)
            self.classes.append(newclass)
            self.database.CreateNewClass(newclass)
        '''

        return None

    def Test(self):
        pass


def main():

    # Environment Definitions
    basepath = os.path.dirname(os.path.abspath(__file__))
    resourcespath = '{}/resources'.format(basepath)
    localdatabasefile = '{}/database.json'.format(resourcespath)
    onlinedatabaseconfigfile = '{}/firebase-config.json'.format(resourcespath)
    objectspath = '{}/objects'.format(resourcespath)
    newshot = '{}/takenimage.jpg'.format(resourcespath)

    # Instance Definitions
    pc = PillClassifier()
    pc.localdatabasefile = localdatabasefile
    pc.onlinedatabaseconfigfile = onlinedatabaseconfigfile
    pc.objectspath = objectspath
    pc.newshot = newshot

    # Instance Operation
    pc.ConnectDatabase()
    pc.TakeaShot()
    pc.PostProcessing()


if __name__ == "__main__":
    main()
