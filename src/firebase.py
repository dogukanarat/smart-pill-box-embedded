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


if __name__ == "__main__":

    print("Main Started")

    if(check()):
        print("Error Occured")
        remove()
