import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
df["class"].value_counts()
species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

#plt.figure(figsize=(7,7))
#sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class' )
#plt.show()
#plt.figure(figsize=(7,7))
#sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class' )
#plt.show()

#sns.heatmap(df.iloc[:,:4].corr(), annot=True)
#plt.show()

from sklearn.tree import DecisionTreeClassifier


# X = df[ ["sepallength","sepalwidth"] ]
# y = df.class_value
# model = DecisionTreeClassifier(max_depth=7, random_state=0)
# model.fit(X, y)
# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X.values, y.values, model)
# plt.show()

X = df[ ["petallength","petalwidth"] ]
y = df.class_value
model = DecisionTreeClassifier(max_depth=9, random_state=0)
model.fit(X, y)
from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X.values, y.values, model)
# plt.show()

from dtreeplt import dtreeplt
# dtree = dtreeplt(model=model, feature_names=X.columns, target_names=['setosa','versicolor','virginica'])
# dtree.view()
# plt.show()

#estymator dla 4 cech
X = df.iloc[:, :4]
y = df.class_value

model = DecisionTreeClassifier(max_depth=9, random_state=0)
model.fit(X, y)
dtreeplt(model, feature_names=X.columns, target_names=["setosa","versicolor","virginica"]).view()
plt.show()
print(pd.DataFrame(model.feature_importances_, X.columns))
print(pd.DataFrame(model.feature_importances_, X.columns).sum())