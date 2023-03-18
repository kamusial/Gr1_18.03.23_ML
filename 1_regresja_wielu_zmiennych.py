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

sns.displot(df.Weight)
plt.show()


