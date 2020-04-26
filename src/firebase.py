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
        "user_name": "Doğukan",
        "user_unique_id": "asdadasdasd",
        "is_admin": True
    }

    # Pass the user's idToken to the push method
    #results = db.child("Users").push(data)

    all_users = db.child("Users").get()
    for user in all_users.each():
        print(user.key())
        print(user.val())
        print(type(user.val()))

    result = db.child("StatusParameters").get()

    data = {'class_name': 'SampleNameA',
            'sample_path': 'd:\\Github\\smart-pill-box-embedded\\src/resources/samples/86fa6043-1d77-4c1c-a281-359bacdc68b3.png',
            'sample_amount': 2,
            'feature_vector': [[1370, 1248, 1246, 2944,   52],
                               [1418, 1304, 1299, 2966,   51]],
            'unique_class_name': '86fa6043-1d77-4c1c-a281-359bacdc68b3'
            }

    print(type(data))

    #result = db.child("PillClasses").push(data)
