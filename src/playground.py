from datetime import datetime, date, time, timezone, timedelta

frequency = 2
delta = timedelta(hours=frequency)

print(type(delta))
print(int(delta.seconds/3600))
