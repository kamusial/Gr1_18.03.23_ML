import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('weight-height.csv')
print(type(df))
#print(df)
print(df.head(10))
print('\n')

#loc - nazwy, labelki
#iloc - inxeksy, pozycje
# print(df.iloc[:,1])
# print(df.iloc[[3, 4],[1,2]])
# print(df.loc[:,['Height','Weight']])

print(df.Gender.value_counts())   #zliczanie wartości
#klasy są zbalansowane, po 5000
df.Height *= 2.54
df.Weight /= 2.2
print(df.head(10))

# sns.displot(df.Weight)
# plt.show()
# razem dla male i female

# sns.displot(df.query("Gender=='Male'").Weight, stat='density')
# plt.title('Male')
# sns.displot(df.query("Gender=='Female'").Weight)
# plt.title('Famale')
sns.displot(df, x='Weight', hue='Gender')
#plt.show()

#gender nie jest daną numeryczną
df = pd.get_dummies(df)   #zmiana na dane numeryczne
print(df)
del(df["Gender_Male"])
print(df)
df = df.rename(columns={"Gender_Female":"Gender"})
print(df)
# 1 - female,  0 - male

model = LinearRegression()
model.fit(df[["Height","Gender"]],df["Weight"])  #wejście, wynik
print(model.coef_, model.intercept_)
df2 = pd.DataFrame(model.coef_, ["Height","Gender"])
print(df2)

#moje dane
gender = 1
height = 192
weight = model.intercept_ + model.coef_[0]*height + model.coef_[1]*gender
print(weight)

