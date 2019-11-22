import asyncio
import time
from contextlib import suppress
import pyrebase

config = {
    "apiKey": "AIzaSyBCzObI1ul0sB61TIi_XA83vpmsi30DGJQ",
    "authDomain": "pill-classification.firebaseapp.com",
    "databaseURL": "https://pill-classification.firebaseio.com",
    "storageBucket": "pill-classification.appspot.com"
}

firebaseKernel = pyrebase.initialize_app(config)
firebase = firebaseKernel.database()


def check():
    return firebase.child("statusParams").child("errorOccured").get().val()


def remove():
    firebase.child("statusParams").child("errorOccured").set(False)
    return None


async def stateFirst():
    while True:
        print("Executed")
        if(check()):
            print("Error Occured")
            remove()
        await asyncio.sleep(1)


async def echo_forever():
    while True:
        print("echo")
        await asyncio.sleep(1)


def printArray():
    print([i for i in range(99)])
    return None


async def main():
    asyncio.ensure_future(stateFirst())  # fire and forget

    printArray()
    await asyncio.sleep(1)
    printArray()
    await asyncio.sleep(1)
    printArray()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(main())
        loop.run_forever()

    except KeyboardInterrupt:
        pass

    finally:
        print("Closing Loop")
        loop.close()
