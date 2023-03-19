import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
print(df['class'].value_counts())
#klasy zbalansowane

#class - zmienna wynikowa powinna mieć charakter numeryczny
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)
#można replace, ale map zachowa oryginalną kolumnę
print(df["class_value"].value_counts())

# print(df.describe())

#nowy kwiat - moj
sample = np.array([5.6, 3.2, 5.2, 1.45])

# #plt.figure(figsize=(7,7))  #ustawienie wielkości wykresu
# sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
# plt.scatter(5.6, 3.2, c='r')
# #plt.show()
#
# sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
# plt.scatter(5.2, 1.45, c='r')
# #plt.show()

#obliczanie dystansu pomiędzy probką, a istniejącymi danymi
df["distance"] = (df.sepallength-sample[0])**2 + (df.sepalwidth-sample[1])**2 +\
                 (df.petallength-sample[2])**2 + (df.petalwidth-sample[3])**2
print(df.sort_values("distance"))
print(df.sort_values("distance").head(3)['class'].value_counts())
print(df.sort_values("distance").head(5)['class'].value_counts())

print(df.head(5).to_string())
X = df.iloc[:, :4]
y = df.class_value   #dane_wynikowe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
#algorytm knn w wersji klasyfikatora sklearn
model = KNeighborsClassifier(5)    #liczba sąsiadów
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test)  ) ))

result = []
for k in range(1, 100):
    model = KNeighborsClassifier(k)  # liczba sąsiadów
    model.fit(X_train, y_train)
    result.append(model.score(X_test, y_test))
plt.plot(range(1,100), result)
plt.grid()
plt.show()

#zapis i załadowanie modelu
import joblib
# model = KNeighborsClassifier(5)
# model.fit(X_train, y_train)
#model.score(X_test, y_test)
#joblib.dump(model, 'knn.model')   #zapis modelu jako plik
model_new = joblib.load('knn.model')
model_new.predict(sample.reshape(1, -1))
print(dir(model_new))
print(model_new.n_neighbors)



