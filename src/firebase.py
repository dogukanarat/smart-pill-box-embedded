import pyrebase

config = {
    "apiKey": "AIzaSyBCzObI1ul0sB61TIi_XA83vpmsi30DGJQ",
    "authDomain": "pill-classification.firebaseapp.com",
    "databaseURL": "https://pill-classification.firebaseio.com",
    "storageBucket": "pill-classification.appspot.com"
}

firebaseKernel = pyrebase.initialize_app(config)
db = firebaseKernel.database()
auth = firebaseKernel.auth()


def check():
    return db.child("statusParams").child("errorOccured").get().val()


def remove():
    db.child("statusParams").child("errorOccured").set(False)
    return None


if __name__ == "__main__":

    email = "aratdogukan@gmail.com"
    password = "113412654"

    # Log the user in
    user = auth.sign_in_with_email_and_password(email, password)

    # data to save
    data = {
        "name": "Doğukan"
    }

    # Pass the user's idToken to the push method
    results = db.child("Users").push(data)

    all_users = db.child("Users").get()
    for user in all_users.each():
        print(user.key())
        print(user.val())
        print(type(user.val()))
