import pandas as pd

# Beispiel DataFrame erstellen
data = {'Spalte_A': [1, 2, 3, 4],
        'Spalte_B': [5, 6, 7, 8]}
df = pd.DataFrame(data)

# Benutzerdefinierte Funktion, die auf zwei Spalten angewendet wird
def meine_funktion(row):
    # Hier kannst du deine gewünschte Logik implementieren
    return row['Spalte_A'] + row['Spalte_B']

# Die Funktion mit df.apply auf die beiden Spalten anwenden
df['Ergebnis'] = df.apply(meine_funktion, axis=1)

# Ergebnis anzeigen
print(df)
