from datetime import date, timedelta,datetime


today  = datetime.now()


time_start = today.strftime('%d.%m.%Y %H:00')
print(time_start)