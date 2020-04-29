from embedded import PillClassifier
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


async def routine_database_update():
    while True:
        await asyncio.sleep(1)

        if(pc.check_database()):
            print("Database Update Trigger!")
            pc.fetch_content()


async def routine_new_pill():
    while True:
        await asyncio.sleep(1)

        if(pc.check_new_pill_cmd()):
            print("New Pill Trigger!")
            pc.take_shot()
            pc.post_processing()
            pc.push_content()


async def routine_last_take():
    while True:
        await asyncio.sleep(5)

        pc.check_last_take()
        print("Last Take Checked!")


async def for_demo_patient_came():
    await asyncio.sleep(10)
    user_key = "QXDGIpt0rUPSFmJqD6dPxIk2Qog1"
    print("Patient came!")
    pc.set_last_take(user_key)


if __name__ == "__main__":

    loop = asyncio.get_event_loop()

    try:
        asyncio.ensure_future(routine_database_update())
        asyncio.ensure_future(routine_new_pill())
        asyncio.ensure_future(routine_last_take())
        # asyncio.ensure_future(for_demo_patient_came())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Operation is aborted!")
        loop.close()
