import tracemalloc
import random
import datetime
import os
from multiprocessing import Process
import asyncio
import time
import pyrebase
import sys
from embedded import *

config = {
    "apiKey": "AIzaSyBCzObI1ul0sB61TIi_XA83vpmsi30DGJQ",
    "authDomain": "pill-classification.firebaseapp.com",
    "databaseURL": "https://pill-classification.firebaseio.com",
    "storageBucket": "pill-classification.appspot.com"
}

firebaseKernel = pyrebase.initialize_app(config)
firebase = firebaseKernel.database()


tracemalloc.start()


def background():
    while True:
        pass


def main():
    embedded()
    '''
    while True:
        pass
    '''


if __name__ == '__main__':
    proc_background = Process(target=background)
    proc_main = Process(target=main)
    proc_background.start()
    proc_main.start()
