import pandas as pd

# Startdatum und Enddatum festlegen
start_date = '2023-01-01'
end_date = '2023-01-10'

# Erstellen Sie einen Datumsbereich mit Pandas
date_range = pd.date_range(start=start_date, end=end_date)

# Iterieren Sie über den Datumsbereich mit einer for-Schleife
for date in date_range:
    print(date.strftime('%Y-%m-%d'))