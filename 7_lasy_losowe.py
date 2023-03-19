import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#X = df[ ["sepallength", "sepalwidth"] ]   #słabe cechy
X = df[ ["petallength", "petalwidth"] ]    #dobre cechy
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

plot_decision_regions(X_train.values, y_train.values, model)
plt.show()

#Konfrontacja z regresją logistyczną
print('Konfrontacja z regresją logistyczną')
df = pd.read_csv("cukrzyca.csv")

X = df.iloc[: , :-1]
y = df.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = RandomForestClassifier(n_estimators=30, max_depth=13, random_state=0)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)) ))