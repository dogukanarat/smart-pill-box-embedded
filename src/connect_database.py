
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
