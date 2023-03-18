import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
print(df.describe().to_string())

print(df.isna().sum())   #brak wartości
print(df.outcome.value_counts())   #na ile zbalansowane są klasy
print(df.columns)

for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)
    mean_ = df[col].mean()
    df[col].replace(np.NaN, mean_, inplace=True)

print(df.isna().sum())
