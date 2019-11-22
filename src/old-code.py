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

# warnings.filterwarnings("ignore")

class JSONDatabase:

    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATABASE_FILE = None
        self.FIREBASE_CONFIG_FILE = None
        self.MAX_CLASS_AMOUNT = 3
        self.MAX_USER_AMOUT = None
        self.databaseContent = None
        self.errors = None
        self.firebase = None

    def Initilize(self):
        try:
            firebaseKernel = pyrebase.initialize_app(self.FIREBASE_CONFIG_FILE)
            self.firebase = firebaseKernel.database()

            if os.path.exists(self.DATABASE_FILE_PATH):

                onlineDatabaseContent = self.GetOnlineDatabase()
                localDatabaseContent = self.GetLocalDatabase()

                if onlineDatabaseContent["UpdateTime"] == localDatabaseContent["UpdateTime"]:
                    self.databaseContent = localDatabaseContent
                    print("Database: Database is loaded!")
                else:
                    self.databaseContent = localDatabaseContent
                    self.UpdateOnlineDatabase()
                    print("Database: Databases are not match!")

                return True
            else:
                currentTime = self.GenerateUpdateTime()
                self.CreateLocalDatabase(time = currentTime)
                self.StoreDatabase()
                print("Database: JSON Database is not found and created new one!")
                return True
        except:
            print("Database: Error occured while initializing the database")
            return False

    def CreateLocalDatabase(self, time=None, updateContent=True):
        if time == None:
            time = self.GenerateUpdateTime()

        self.databaseModel = {
                    "UpdateTime": time,
                    "Classes": [],
                    "Users": [],
                    "StatusParameters": {
                        "IsErrorOccured": False,
                        "Errors": [],
                        "NumberOfUser": 0,
                        "NumberOfClass": 0,
                        "NumberOfPill": [],
                        "PowerMode": "None",
                        "BatteryStatus": "None"
                    }
                }
        if updateContent:
            self.databaseContent = self.databaseModel

        try:
            json.dump(self.databaseModel, codecs.open(self.DATABASE_FILE_PATH, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4)
            return True
        except:
            print("Error occured while creating local database")
            return False
    
    def GenerateUpdateTime(self, printTime=False):
        now = datetime.now()
        currentTime = now.strftime("%c")

        if printTime:
            print(currentTime)

        return currentTime

    def GetLocalDatabase(self):
        try:
            with open(self.DATABASE_FILE_PATH, 'r') as file:
                fileContent = json.loads(file.read())
                return fileContent
        except:
            print("Database: Error occured while getting local database!")
            return False

    def GetOnlineDatabase(self):
        try:
            return self.firebase.get().val()
        except:
            print("Database: Error occured while getting online database!")
            return False

    def StoreDatabase(self):
        try:
            self.UpdateLocalDatabase()
            #self.UpdateOnlineDatabase()
            print("Database: Database is saved!")
            return True
        except:
            print("Database: Error occured while saving database!")
            return False

    def UpdateOnlineDatabase(self):
        try:
            self.firebase.set(self.databaseContent)
            return True
        except:
            print("Database: Error occured while updating online database!")

    def UpdateLocalDatabase(self):
        try:
            json.dump(self.databaseContent, codecs.open(self.DATABASE_FILE_PATH, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4)
            return True
        except:
            print("Database: Error occured while updating local database!")

    def NewClass(self, fileName, objectAmount):
        try:
            numberOfClass = self.GetStatusParameter("NumberOfClass")
            newClassName = "Class{}".format(str(chr(ord('A') + numberOfClass)))
            newClass = {
                    "ClassName": newClassName,
                    "ObjectFileName": fileName,
                    "ObjectAmount": objectAmount
                    }
            self.databaseContent["Classes"].append(newClass)

            newAmountParameter = numberOfClass + 1
            self.UpdateStatusParameter("NumberOfClass", newAmountParameter)

            print("Database: New class is recorded!")
            return True
        except Exception as e:
            print("Database: Error occured while creating new class!")
            print(e)
            return False

    def NewUser(self):
        pass

    def NewPillPeriod(self):
        pass

    def UpdateStatusParameter(self, parameterName, parameterValue):
        self.databaseContent["StatusParameters"][parameterName] = parameterValue
        self.StoreDatabase()
        pass

    def GetStatusParameter(self, parameterName):
        return self.databaseContent["StatusParameters"][parameterName]

    def GetClassParameter(self, classParameter):
        for classObject in self.databaseContent["Classes"]:
            yield classObject[classParameter]

    def GetClasses(self, className=None, classParameter=None):

        response = None

        if classParameter == None:
            if className == None:
                response = self.databaseContent["Classes"]
            else:
                for classObject in self.databaseContent["Classes"]:
                    if (classObject["ClassName"] == className):
                        response = classObject
        else:
            if className == None:
                response = list(self.GetClassParameter(classParameter))
            else:
                for classObject in self.databaseContent["Classes"]:
                    if classObject["ClassName"] == className:
                        response = classObject[classParameter]

        return response

    def GetUsers(self, username):
        pass            

    def CheckStatusError(self):
        status=self.GetStatusParameter("IsErrorOccured")

        if(status):
            self.errors=self.GetStatusParameter("Errors")
            self.UpdateStatusParameter("IsErrorOccured", False)
            return True
        else:
            return False

class PillClassifier:

    def __init__(self):

        # Paths and Files Definitions
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.RESOURCE_PATH = '{}/resources'.format(self.BASE_DIR)
        self.OBJECTS_PATH = '{}/database/objects'.format(self.BASE_DIR)
        self.LOCAL_DATABASE_FILE = '{}.database.json'.format(self.RESOURCE_PATH)
        self.FIREBASE_CONFIG_FILE = '{}.firebase-config.json'.format(self.RESOURCE_PATH)
        self.NEW_OBJECT_FILE = '{}/takenimage.jpg'.format(self.RESOURCE_PATH)

        # Classifier Parameters
        self.SIMILARITY_TRESHOLD = 15084
        self.RESIZE_FACTOR = 0.2
        self.BINS=30
        self.newImage = None
        self.objects = None
        self.objectAmount = None

        # Database Connection
        self.DB = JSONDatabase()
        self.DB.DATABASE_FILE = self.LOCAL_DATABASE_FILE
        self.DB.FIREBASE_CONFIG_FILE = self.FIREBASE_CONFIG_FILE

        self.DB.Initilize()
        self.Initialize()

    def Initialize(self):
        try:
            if not self.DB.GetStatusParameter["NumberOfClass"] == 0:

                storedObjectFiles = glob(os.path.join("{}/*.png".format(self.OBJECTS_PATH)))
                for object in storedObjectFiles:
                    self.objects = cv2.imread(storedObjectFiles)
                
                print("Classifier: Stored objects are imported!")
                return True
            
            else:
                self.objects = None

                print("Classifier: No stored object is found!")
                return True

        except:
            print("Classifier: Error occured while initializing Classifier!")
            return False
    
    def TakeaShot(self):
        try:
            def IsAllSame(objects):
                size, _, _, _ = objects.shape
                result = True

                for pair in itertools.product([i for i in range(size)], repeat=2):
                    pairIter = iter(pair)
                    index1 = next(pairIter)
                    index2 = next(pairIter)
                    histogram1 = self.GetIntensityHistogram(objects[index1])
                    histogram2 = self.GetIntensityHistogram(objects[index2])
                    currentResult = self.CompareHistograms(histogram1, histogram2)

                    if (currentResult < self.SIMILARITY_TRESHOLD):
                        result = False

                return result

            def ResizeImage(image):
                return cv2.resize(image, None, fx = self.RESIZE_FACTOR, fy = self.RESIZE_FACTOR,interpolation = cv2.INTER_AREA)

            def SplitBlobs(image, lowerTreshold, upperTreshold):
                if lowerTreshold <= 10:
                    lowerTreshold = 10

                if upperTreshold >= 255:
                    upperTreshold = 255

                tempImage = np.zeros_like(image)
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        if image[i][j] >= lowerTreshold and image[i][j] <= upperTreshold:
                            tempImage[i][j] = 255
                        else:
                            tempImage[i][j] = 0

                return tempImage

            def MaskObject(object, mask):
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                return object[cY-100:cY+100, cX-100:cX+100]

            def SplitObjects(objects, objectMasks):
                objects = np.array([MaskObject(object, mask) for mask, object in zip(objectMasks, objects)])
                return objects

            def ExtractObjects(image):
                objectAmount = None
                objectMasks = None
                objects = None

                # Step: 1
                grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gaussian = cv2.GaussianBlur(grayscaled, (7, 7), 0)
                unsharped = cv2.addWeighted(grayscaled, 2, gaussian, -1, 0)
                edged = cv2.Canny(unsharped, 70, 800)

                # Step: 2
                edged = 255 - edged
                _, threshed = cv2.threshold(edged, 250, 255, cv2.THRESH_BINARY_INV)
                floodfill = threshed.copy()
                h, w = threshed.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(floodfill, mask, (0, 0), 255)
                inverted = cv2.bitwise_not(floodfill)
                mask = threshed | inverted

                # Step: 3
                _, markers = cv2.connectedComponents(mask)
                objectAmount = np.max(markers)
                labeled = np.uint8(255 * markers / objectAmount)
                markerRange = int(255 / objectAmount)

                objectMasks = np.array([SplitBlobs( labeled, (i+1)*markerRange - 50, (i+1)*markerRange + 50) for i in range(objectAmount)])

                objects = np.array([cv2.bitwise_and(image, image, mask=mask) for mask in objectMasks])

                return (objects, objectAmount)

            # Raspberry Pi Take a Short Function Here!

            newImage = cv2.imread(self.NEW_OBJECT_FILE)
            objects, objectAmount = ExtractObjects(newImage)

            if IsAllSame(objects):
                self.newObjects = objects
                print("Classifier: New image is taken!")
                return True
            else:
                print("Classifier: Objects are not same!")
                return False
            
        except:
            print("Classifier: Error occured while taking new image!")
            return False

    def IsClassExisted(self, histogram):
        if self.GetStatusParameter("NumberOfClass") == 0:
            print("Database: There is no class!")
            return False
        
        histogramPath = "{}/histograms/".format(DATABASE_PATH)
        maximumDistance = 1
        threshold = PillClassifier.SIMILARITY_TRESHOLD
        function = PillClassifier.CompareHistograms
        distances = []

        for objectFileName in self.GetClasses(classParameter="ObjectFileName"):

            image = cv2.imread('{}/objects/{}.png'.format(DATABASE_PATH, objectFileName))

            currentDistance = function(histogram, histogramExisted)

            distances.append(currentDistance)

        distance = min(distances)

        if(distance < threshold):
            thereis = True
        else:
            thereis = False

        if not thereis:
            print("Database: The class is not found!")
            return False
        else:
            print("Database: The class exists!")
            return True

    def DisplayObject(self, objectIndex):
        cv2.imshow('image', self.objects[objectIndex])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    @staticmethod
    def CompareHistograms(firstHistogram, secondHistogram):
        
        result = cdist(firstHistogram.reshape(-1, 1).transpose(),
                       secondHistogram.reshape(-1, 1).transpose(), 'cityblock')
        return result[0][0]
        '''
        result = wasserstein_distance(firstHistogram.reshape(-1, 1).transpose(),
                       secondHistogram.reshape(-1, 1).transpose())
        return result
        '''

    @staticmethod
    def GetIntensityHistogram(image, plot=False):

        RGBHistogram = []

        color = ('b','g','r')
        for channel, col in enumerate(color):
            histogram = cv2.calcHist([image],[channel],None,[256],[0,256])
            RGBHistogram.append(histogram)
            plt.plot(histogram,color = col)
            plt.xlim([0,256])
        
        if(plot):
            plt.title('Histogram for the image')
            plt.show()

        return np.array(RGBHistogram)
    
    @staticmethod
    def GetOrientationHistogram(image, plot=False):
        '''
        object = self.objects[objectIndex]

        histogram= hog(object, orientations=2, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), multichannel=True)
        
        # Find the way to convert list to histogram 
        print(np.array(histogram).shape)
        
        if(plot):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            ax1.axis('off')

            ax1.imshow(object, cmap=plt.cm.gray)
            ax1.set_title('Input Image')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(histogram, in_range=(0, 10))

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
        '''
        pass

    def SaveObjectToDatabase(self, objectIndex, fileName="None", objectAmount=0):
        try:
            self.DB.NewClass(fileName, histogramFileName, objectAmount)
            self.DB.StoreDatabase()
            return True
        except:
            return False
        
    def SaveObject(self, objectIndex):

        object = self.objects[objectIndex]
        histogram = self.GetIntensityHistogram(objectIndex)

        if not self.DB.IsClassExisted(histogram):
            uniqueFileName = str(uuid.uuid4())

            cv2.imwrite("{}/{}.png".format(self.OBJECTS_PATH,uniqueFileName), object)

            result = self.SaveObjectToDatabase(objectIndex, fileName=uniqueFileName, objectAmount=0)

            if result:
                print("Classifier: The object is saved to the database")
                return True
            else:
                print("Classifier: Error occured while saving the object")
                return True
            
        else:
            print("Classifier: The object is already existed in the database")
            return False

def main():
    Classifier = PillClassifier()

if __name__ == "__main__":
    main()
