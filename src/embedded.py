from PillClassifierBackend import PillClassifierBackend
from datetime import datetime, date, time, timezone, timedelta
from glob import glob
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_closing
from scipy.stats import wasserstein_distance
from skimage.feature import hog
from skimage import data, exposure
from datetime import datetime
from types import SimpleNamespace
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
import sys
import time


class User():
    AMOUNT = 0
    INSTANCES = []

    def __init__(self, user_name=None, user_unique_id=None, is_admin=None, pill_periods=None):
        self.user_name = user_name
        self.user_unique_id = user_unique_id
        self.is_admin = is_admin
        self.pill_periods = pill_periods

        User.AMOUNT += 1
        User.INSTANCES.append(weakref.proxy(self))

    def get_dictionary(self):
        return {
            "user_name": self.user_name,
            "user_unique_id": self.user_unique_id,
            "is_admin": self.is_admin,
            "pill_periods": [period.get_dictionary() for period in self.pill_periods]
        }

    def set_dictionary(self, dictionary):
        variables = SimpleNamespace(**dictionary)
        self.user_name = variables.user_name
        self.user_unique_id = variables.user_unique_id
        self.is_admin = variables.is_admin

        if "pill_periods" in dictionary:
            for period in variables.pill_periods:
                new_period = PillPeriod()
                new_period.set_dictionary(period)
                self.pill_periods.append(new_period)

    def set_pill_period(self, pill_period):
        self.pill_periods.append(pill_period)


class PillClass():
    AMOUNT = 0
    INSTANCES = []

    def __init__(self, class_name=None, sample_path=None, sample_amount=None, feature_vector=None, unique_class_name=None):
        self.class_name = class_name
        self.sample_path = sample_path
        self.sample_amount = sample_amount
        self.unique_class_name = unique_class_name
        self.feature_vector = feature_vector

        PillClass.AMOUNT += 1
        PillClass.INSTANCES.append(weakref.proxy(self))

    def get_dictionary(self):
        return {
            "class_name": self.class_name,
            "sample_path": self.sample_path,
            "sample_amount": self.sample_amount,
            "feature_vector": self.feature_vector.tolist(),
            "unique_class_name": self.unique_class_name
        }

    def set_dictionary(self, dictionary):
        variables = SimpleNamespace(**dictionary)
        self.class_name = variables.class_name
        self.sample_path = variables.sample_path
        self.sample_amount = variables.sample_amount
        self.feature_vector = variables.feature_vector
        self.unique_class_name = variables.unique_class_name

    def set_new_samples(self, sample_amount):
        self.sample_amount += sample_amount
        return None


class PillPeriod():
    AMOUNT = 0
    INSTANCES = []

    def __init__(self, class_name=None, frequency=0, last_take=None):

        self.class_name = class_name
        self.frequency = timedelta(seconds=frequency)
        self.last_take = last_take

        PillPeriod.AMOUNT += 1
        PillPeriod.INSTANCES.append(weakref.proxy(self))

    def get_dictionary(self):
        return {
            "class_name": self.class_name,
            "frequency": self.frequency,
            "last_take": self.last_take,
        }

    def set_dictionary(self, dictionary):
        variables = SimpleNamespace(**dictionary)
        self.class_name = variables.class_name
        self.frequency = variables.frequency
        self.last_take = variables.last_take

    def if_passed(self):
        if(datetime.now() >= self.frequency + self.last_take):
            print("Passed!")
            return True
        else:
            print("NOT Passed!")
            return False

    def set_last_take(self):
        self.last_take = datetime.now()
        return None


class Database():

    def __init__(self, local_database_file, online_database_config_file, objects_path):
        self.local_database_file = local_database_file
        self.online_database_config_file = online_database_config_file
        self.objects_path = objects_path

        self.content = None
        self.firebase_db = None
        self.firebase_auth = None

        self.status_parameters = []
        self.users = []
        self.pill_classes = []
        self.pill_periods = []

        self.initialize()

    def initialize(self):
        try:
            self.get_database_content()

            self.status_parameters = self.content['StatusParameters']

            if 'Classes' in self.content:
                for pill_class in self.content['Classes']:
                    pill_class_object = PillClass()
                    pill_class_object.set_dictionary(
                        self.content["Classes"][pill_class])
                    self.pill_classes.append(pill_class_object)

            if 'Users' in self.content:
                for user in self.content['Users']:
                    user_object = User()
                    user_object.set_dictionary(self.content["Users"][user])
                    self.users.append(user_object)

            return True

        except Exception as e:
            print(e)
            return False

    def get_database_content(self):
        try:
            local_database_content = self.get_local_database()
            online_database_content = dict(self.get_online_database())

            '''
            if local_database_content["UpdateTime"] == online_database_content["UpdateTime"]:
                self.content = localdatabasecontent
                return True
            else:
                return False
            '''

            self.content = online_database_content

            return True

        except Exception as e:
            print(e)
            return False

    def get_local_database(self):
        try:
            with open(self.local_database_file, 'r') as file:
                return json.loads(file.read())

        except Exception as e:
            print(e)
            return False

    def get_online_database(self):
        try:
            with open(self.online_database_config_file, 'r') as file:
                file_content = json.loads(file.read())
                firebase_kernel = pyrebase.initialize_app(file_content)
                self.firebase_db = firebase_kernel.database()
                self.firebase_auth = firebase_kernel.auth()

            return self.firebase_db.get().val()

        except Exception as e:
            print(e)
            return False

    def set_new_pill_class(self, pill_class_object):

        self.pill_classes.append(pill_class_object)

        results = self.firebase_db.child("Classes").push(
            pill_class_object.get_dictionary())

        return True

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

    def debug(self, message):
        print("-----------------------DEBUG-----------------------")
        print(message)
        print("-----------------------DEBUG-----------------------")


class PillClassifier():

    def __init__(self):
        self.backend = PillClassifierBackend()
        self.objects_path = None
        self.local_database_file = None
        self.online_database_config_file = None
        self.new_shot_file = None
        self.samples_path = None
        self.database = None
        self.users = []
        self.pill_classes = []
        self.temp_object_list = []
        self.total_sample_amount = 0
        self.total_pill_class_amount = 0

    def initialize(self):
        try:
            self.database = Database(
                self.local_database_file, self.online_database_config_file, self.objects_path)

            self.users = self.database.users
            self.pill_classes = self.database.pill_classes

            return True

        except Exception as e:
            print(e)
            return False

    def take_shot(self):
        try:
            new_image = cv2.imread(self.new_shot_file)
            object_list, object_amount = self.backend.extract_objects(
                new_image)
            self.temp_object_list = object_list

            print("Classifier: Taking a shot is complete!")
            return True

        except:
            print("Classifier: Error occured while taking new image!")
            print(sys.exc_info()[0])
            return False

    def load_classifier(self):
        return None

    def post_processing(self):
        ''' Temprorary '''
        self.temp_object_list = [
            self.temp_object_list[3], self.temp_object_list[6]]
        ''' Temprorary '''

        feature_vector = self.backend.is_all_same(self.temp_object_list)
        sample_amount = len(self.temp_object_list)

        if feature_vector is None:
            print("Classifier: Objects are NOT same!")
            return None

        if self.if_pill_class_exist(feature_vector):
            self.set_new_samples(feature_vector, sample_amount)
            print(
                f"Classifier: Pill class does exist! Added {sample_amount} more pills to the class!")
        else:

            if (self.total_pill_class_amount >= 3):
                print(
                    f"Classifier: Pill class does NOT exist! However max pill class CANNOT be exceeded!")
                return None

            self.set_new_pill_class(feature_vector)
            print("Classifier: Pill class does NOT exist! Created new pill class!")

        self.temp_object_list = []

        return None

    def set_new_samples(self, feature_vector, sample_amount):
        releated_pill_class = self.get_pill_class(feature_vector)
        releated_pill_class.set_new_samples(sample_amount)
        self.total_sample_amount += sample_amount

    def set_new_pill_class(self, feature_vector):

        new_pill_class_name = f"SampleName{chr(65+len(self.pill_classes))}"
        new_pill_class_feature_vector = feature_vector
        new_pill_class_sample_amount = len(self.temp_object_list)
        new_pill_class_unique_id = self.generate_unique_id()
        new_pill_class_sample_path = self.store_sample_image(
            self.temp_object_list[0], new_pill_class_unique_id)

        new_pill_class = PillClass(
            class_name=new_pill_class_name,
            sample_path=new_pill_class_sample_path,
            sample_amount=new_pill_class_sample_amount,
            feature_vector=new_pill_class_feature_vector,
            unique_class_name=new_pill_class_unique_id
        )

        self.total_pill_class_amount += 1
        self.total_sample_amount += new_pill_class_sample_amount
        self.pill_classes.append(new_pill_class)

        self.database.set_new_pill_class(new_pill_class)

        print("Classifier: New class added")

    def if_pill_class_exist(self, feature_vector):
        result = False

        for pill_class in self.pill_classes:

            local_vector = pill_class.feature_vector
            result, distance = self.backend.is_all_similar(
                np.array([feature_vector, local_vector]))

        return result

    def generate_unique_id(self):
        return str(uuid.uuid4())

    def store_sample_image(self, sample_image, unique_class_name):
        try:
            cv2.imwrite(
                f"{self.samples_path}/{unique_class_name}.png",
                sample_image
            )

            print(
                f"Classifier: New sample stored at {self.samples_path}/{unique_class_name}.png"
            )

            return f"{self.samples_path}/{unique_class_name}.png"

        except Exception as e:
            print("Classifier: Error occured while storing sample image!")
            print(e)
            return False

    def get_pill_class(self, feature_vector):
        for index, pill_class in enumerate(self.pill_classes):
            if (pill_class.feature_vector == feature_vector).all():
                return self.pill_classes[index]

        print("Classifier: The feature vector could NOT be found in the list!")
        return False

    def debug(self, message):
        print("-----------------------DEBUG-----------------------")
        print(message)
        print("-----------------------DEBUG-----------------------")

    def info(self, message):
        print("-----------------------INFO-----------------------")
        print(message)
        print("-----------------------INFO-----------------------")


def embedded():

    # Environment Definitions
    base_path = os.path.dirname(os.path.abspath(__file__))
    resources_path = f'{base_path}/resources'

    # Instance Definitions
    pc = PillClassifier()
    pc.local_database_file = f'{resources_path}/database.json'
    pc.online_database_config_file = f'{resources_path}/firebase-config.json'
    pc.samples_path = f'{resources_path}/samples'
    pc.new_shot_file = f'{resources_path}/real_image.jpg'

    pc.initialize()

    print(f'Number of class: {len(pc.pill_classes)}')
    print(f'Number of user: {len(pc.users)}')

    pc.take_shot()
    pc.post_processing()


if __name__ == "__main__":
    embedded()
