from PillClassifierBackend import PillClassifierBackend
from datetime import datetime, date, time, timezone, timedelta
from glob import glob
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import binary_closing
from scipy.stats import wasserstein_distance
from skimage.feature import hog
from skimage import data, exposure
from types import SimpleNamespace
import traceback
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

    def __init__(self, key=None, user_name=None, user_unique_id=None, is_admin=None, pill_periods=None):
        self.key = key
        self.user_name = user_name
        self.user_unique_id = user_unique_id
        self.is_admin = is_admin

        User.AMOUNT += 1

    def __del__(self):
        User.AMOUNT -= 1

    def get_dictionary(self):
        return {
            f"Users/{self.key}/": {
                "user_name": self.user_name,
                "user_unique_id": self.user_unique_id,
                "is_admin": self.is_admin
            }
        }

    def set_dictionary(self, dictionary, key):
        variables = SimpleNamespace(**dictionary)
        self.key = key
        self.user_name = variables.user_name
        self.user_unique_id = variables.user_unique_id
        self.is_admin = variables.is_admin


class PillClass():
    AMOUNT = 0

    def __init__(self, key=None, class_name=None, sample_path=None, sample_amount=None, feature_vector=None, unique_class_name=None):
        self.key = key
        self.class_name = class_name
        self.sample_path = sample_path
        self.sample_amount = sample_amount
        self.unique_class_name = unique_class_name
        self.feature_vector = feature_vector

        PillClass.AMOUNT += 1

    def __del__(self):
        PillClass.AMOUNT -= 1

    def get_dictionary(self):
        return {
            f"Classes/{self.key}/": {
                "class_name": self.class_name,
                "sample_path": self.sample_path,
                "sample_amount": self.sample_amount,
                "feature_vector": self.feature_vector,
                "unique_class_name": self.unique_class_name
            }
        }

    def set_dictionary(self, dictionary, key):
        variables = SimpleNamespace(**dictionary)
        self.key = key
        self.class_name = variables.class_name
        self.sample_path = variables.sample_path
        self.sample_amount = variables.sample_amount
        self.feature_vector = variables.feature_vector
        self.unique_class_name = variables.unique_class_name

    def set_new_samples(self, sample_amount):
        self.sample_amount += sample_amount
        return None

    def set_take(self, sample_amount):
        self.sample_amount -= sample_amount


class PillPeriod():
    AMOUNT = 0

    def __init__(self, key=None, message=False, user_name=None, class_name=None, frequency=0, last_take=datetime.now().strftime("%m/%d/%Y %H:%M")):

        self.key = key
        self.user_name = user_name
        self.class_name = class_name
        self.frequency = timedelta(hours=frequency)
        self.last_take = datetime.strptime(last_take, "%m/%d/%Y %H:%M")
        self.message = message

        PillPeriod.AMOUNT += 1

    def __del__(self):
        PillPeriod.AMOUNT -= 1

    def get_dictionary(self):
        return {
            f"Periods/{self.key}/": {
                "user_name": self.user_name,
                "class_name": self.class_name,
                "frequency": self.frequency.seconds/3600,
                "sample_amount": self.sample_amount,
                "last_take": self.last_take.strftime("%m/%d/%Y %H:%M"),
                "message": self.message
            }
        }

    def set_dictionary(self, dictionary, key):
        variables = SimpleNamespace(**dictionary)
        self.key = key
        self.user_name = variables.user_name
        self.class_name = variables.class_name
        self.frequency = timedelta(hours=variables.frequency)
        self.sample_amount = variables.sample_amount
        self.message = variables.message
        if (variables.last_take != None):
            self.last_take = datetime.strptime(
                variables.last_take, "%m/%d/%Y %H:%M")

    def if_passed(self):
        if(datetime.now() >= self.frequency + self.last_take):
            return True
        else:
            return False

    def set_last_take(self):
        self.message = False
        self.last_take = datetime.now()
        return None

    def set_message(self):
        self.message = True
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
                        self.content["Classes"][pill_class], pill_class)
                    self.pill_classes.append(pill_class_object)

            if 'Users' in self.content:
                for user in self.content['Users']:
                    user_object = User()
                    user_object.set_dictionary(
                        self.content["Users"][user], user)
                    self.users.append(user_object)

            if 'Periods' in self.content:
                for period in self.content['Periods']:
                    pill_period_object = PillPeriod()
                    pill_period_object.set_dictionary(
                        self.content["Periods"][period], period)
                    self.pill_periods.append(pill_period_object)

            return True

        except Exception as e:
            print("Database: Error while initializing the database!")
            traceback.print_exc()
            return False

    def get_database_content(self):
        try:
            #local_database_content = self.get_local_database()
            online_database_content = dict(self.get_online_database())

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

        data = pill_class_object.get_dictionary()
        for key in data:
            ref_key = key.split("/")[1]
            results = self.firebase_db.child(
                f"{ref_key}").set(data[key])

        return True

    def set_database_content(self):
        try:

            status_parameters_data = {
                "StatusParameters": {
                    "UserAmount": User.AMOUNT,
                    "ClassAmount": PillClass.AMOUNT,
                    "DatabaseUpdated": False,
                    "IsErrorOccured": self.status_parameters["IsErrorOccured"],
                    "NewPillCmd": False,
                    "PowerMode": "Batter Mode"
                }
            }

            self.set_online_database(status_parameters_data)

            for pill_period in self.pill_periods:
                pill_period_data = pill_period.get_dictionary()
                self.set_online_database(pill_period_data)

            for pill_class in self.pill_classes:
                pill_class_data = pill_class.get_dictionary()
                self.set_online_database(pill_class_data)

            print("Database: Database is saved!")
            return True
        except:
            traceback.print_exc()
            print("Database: Error occured while saving database!")
            return False

    def set_online_database(self, content):
        try:
            self.firebase_db.update(content)
            return True
        except:
            traceback.print_exc()
            print("Database: Error occured while updating online database!")

    def fetch_database(self):
        try:
            self.content.clear()
            self.status_parameters.clear()
            self.users.clear()
            self.pill_classes.clear()
            self.pill_periods.clear()

            self.initialize()

            self.firebase_db.child(
                "StatusParameters").update({"DatabaseUpdated": False})

            return True

        except:
            traceback.print_exc()
            print("Database: Error occured while fetching content!")
            return False

    def if_updated(self):
        return self.firebase_db.child("StatusParameters/DatabaseUpdated").get().val()

    def if_new_pill_cmd(self):
        result = self.firebase_db.child(
            "StatusParameters/NewPillCmd").get().val()
        if(result):
            self.firebase_db.child("StatusParameters").update(
                {"NewPillCmd": False})
            return True
        else:
            return False

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

        self.status_parameters = []
        self.users = []
        self.pill_classes = []
        self.pill_periods = []

        self.temp_object_list = []

        self.total_sample_amount = 0
        self.total_pill_class_amount = 0

    def initialize(self):
        try:
            self.database = Database(
                self.local_database_file, self.online_database_config_file, self.objects_path)

            self.status_parameters = self.database.status_parameters
            self.users = self.database.users
            self.pill_classes = self.database.pill_classes
            self.pill_periods = self.database.pill_periods

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
            self.temp_object_list[4], self.temp_object_list[5]]
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

        unique_id = self.generate_unique_id()

        new_dictionary = {
            "class_name": f"SampleName{chr(65+len(self.pill_classes))}",
            "feature_vector": feature_vector.tolist(),
            "sample_amount": len(self.temp_object_list),
            "unique_class_name": unique_id,
            "sample_path": self.store_sample_image(self.temp_object_list[0], unique_id)
        }

        new_pill_class_key = self.database.firebase_db.child(
            "Classes").generate_key()

        new_pill_class = PillClass()
        new_pill_class.set_dictionary(new_dictionary, new_pill_class_key)

        self.total_pill_class_amount += 1
        self.total_sample_amount += len(self.temp_object_list)
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

    def push_content(self):
        try:
            self.database.status_parameters = self.status_parameters
            self.database.pill_classes = self.pill_classes
            self.database.pill_periods = self.pill_periods

            self.database.set_database_content()

            return True
        except:
            return False

    def fetch_content(self):
        try:
            self.status_parameters.clear()
            self.users.clear()
            self.pill_classes.clear()
            self.pill_periods.clear()

            self.database.fetch_database()
            self.status_parameters = self.database.status_parameters
            self.users = self.database.users
            self.pill_classes = self.database.pill_classes
            self.pill_periods = self.database.pill_periods

            print("Classifier: Fetching complete!")
            return True
        except:
            traceback.print_exc()
            print("Classifier: Error occured while fetching content!")
            return False

    def check_database(self):
        return self.database.if_updated()

    def check_new_pill_cmd(self):
        return self.database.if_new_pill_cmd()

    def check_last_take(self):
        for pill_period in self.pill_periods:
            if(pill_period.if_passed()):
                print(f"Passed: {pill_period.key}")
                # SENT A MESSAGE TO PATIENT
                if(not pill_period.message):
                    pill_period.set_message()
                    self.push_content()
                    self.fetch_content()

    def set_last_take(self, user_key):

        user_name = None
        for user in self.users:
            if(user.user_unique_id == user_key):
                print(f"User Found: {user_name}")
                user_name = user.user_name

        for pill_period in self.pill_periods:
            if(pill_period.user_name == user_name and pill_period.if_passed()):
                pill_period.set_last_take()

                for pill_class in self.pill_classes:
                    if pill_class.class_name == pill_period.class_name:
                        pill_class.set_take(pill_period.sample_amount)

                        if pill_class.sample_amount <= 4:
                            print("AZ KALDI!")
                            self.status_parameters["IsErrorOccured"] = True

                print(f"{pill_period.class_name} is given to the patient!")
                self.push_content()
                self.fetch_content()

        print("Finished!")

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


if __name__ == "__main__":
    pass
