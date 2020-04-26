
class Database():

    def __init__(self, localdatabasefile, onlinedatabaseconfigfile, objectspath):
        self.localdatabasefile = localdatabasefile
        self.onlinedatabaseconfigfile = onlinedatabaseconfigfile
        self.objectspath = objectspath
        self.content = None
        self.firebase = None
        self.users = []
        self.pillclasses = []

    def Initialize(self):
        try:
            self.GetDatabaseContent()

            self.statusparameters = self.content['StatusParameters']

            if bool(self.content['Classes']):
                for pillclass in self.content['Classes']:
                    pillclassobject = PillClass(
                        pillclass["ClassName"],
                        self.objectspath,
                        pillclass["Amount"],
                        uniqueclassname=pillclass["UniqueClassName"])
                    self.pillclasses.append(pillclassobject)

            if bool(self.content['Users']):
                for user in self.content['Users']:

                    pillperiods = []

                    if bool(user['PillPeriods']):
                        for pillperiod in user['PillPeriods']:
                            pillperiodobject = PillPeriod(
                                user["UserUniqueID"],
                                pillperiod["ClassName"],
                                pillperiod["LastTake"]
                            )
                            pillperiods.append(pillperiodobject)

                    userobject = User(
                        user["Username"],
                        user["UserUniqueID"],
                        user["IsAdmin"],
                        pillperiods
                    )

                    self.users.append(userobject)

            return True
        except Exception as e:
            print(e)
            return False

    def GetDatabaseContent(self):
        localdatabasecontent = self.GetLocalDatabase()
        onlinedatabasecontent = dict(self.GetOnlineDatabase())

        if localdatabasecontent["UpdateTime"] == onlinedatabasecontent["UpdateTime"]:
            self.content = localdatabasecontent
            return True
        else:
            return False

    def GetLocalDatabase(self):
        try:
            with open(self.localdatabasefile, 'r') as file:
                return json.loads(file.read())
        except:
            return False

    def GetOnlineDatabase(self):
        try:
            with open(self.onlinedatabaseconfigfile, 'r') as file:
                filecontent = json.loads(file.read())
                firebasekernel = pyrebase.initialize_app(filecontent)
                self.firebase = firebasekernel.database()

            return self.firebase.get().val()
        except:
            return False

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

    def CreateNewUser(self):
        pass

    def CreateNewClass(self, pillclassobject):
        self.pillclasses.append(pillclassobject)
        self.content["Classes"].append(pillclassobject.GetDictionary())
        self.content["StatusParameters"]
        self.SetDatabaseContent()
        pass

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
