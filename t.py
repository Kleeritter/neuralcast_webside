import pandas as pd

# Beispiel DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 5], 'Name': [25, 30, 22]})

# Merge mit unterschiedlichen Suffixen für die Spaltennamen
# Setze die 'ID'-Spalte als Index
df1.set_index('ID', inplace=True)
df2.set_index('ID', inplace=True)

# Merge mit unterschiedlichen Suffixen für die Spaltennamen
merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer', suffixes=('_df1', '_df2'))

# Anzeigen des resultierenden DataFrames
print(merged_df)

