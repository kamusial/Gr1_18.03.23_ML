import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("otodom.csv")
print(df.head())
print(df.describe().to_string())

# print(df.corr())    #korelacja
# sns.heatmap(df.corr(), annot=True)
# plt.show()

#weźmy tylko cechy
# sns.heatmap(df.iloc[:, 2:].corr(), annot=True)  #wyciąłem cenę
# plt.show()
#
# sns.displot(df.cena)
# plt.show()

_min = df.describe().loc["min", "cena"]
q1 = df.describe().loc["25%", "cena"]
q3 = df.describe().loc["75%", "cena"]
print(_min, q1, q3)

df1 = df[(df.cena >= _min) & (df.cena <= q3)]
sns.displot(df1.cena)
plt.show()

X = df1.iloc[:, 2:]   #wyznaczam Xy, bez ID i ceny
y = df1.cena

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)    #model z 80% danych - tylko dane treningowe
print(model.score(X_test, y_test))
#print(model.coef_)
print(pd.DataFrame(model.coef_, X.columns))
print('\n')
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))