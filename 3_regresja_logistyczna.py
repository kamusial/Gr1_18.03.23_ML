import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
print(df.describe().to_string())
df.to_csv('cukrzyca.csv', index=False)

X = df.iloc[: , :-1]
y = df.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('zmiana danych')

print(df.outcome.value_counts())
df1 = df.query(" outcome==0 ").sample(n=500, random_state=0)
df2 = df.query(" outcome==1 ").sample(n=500, random_state=0)
df3 = pd.concat([df1, df2])
X = df3.iloc[: , :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))