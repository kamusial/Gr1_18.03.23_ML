import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv', comment='#')
print(df.head(10).to_string())
print(df.target.value_counts())  #rozklad wartosci

X = df.iloc[: , :-1]
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

model = DecisionTreeClassifier(max_depth=5, random_state=0, max_features=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

model = DecisionTreeClassifier(max_depth=9, random_state=0, max_features=10, min_samples_split=4)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame( confusion_matrix(y_test, model.predict(X_test) ) ))

from sklearn.model_selection import GridSearchCV

model = DecisionTreeClassifier()
params = {
    "max_depth" : range(3,14),
    "max_features" : range(5, X_train.shape[1]+1, 2),
    "min_samples_split" : [2,4,5],
    "random_state" : [0]
}

grid = GridSearchCV(model, params, scoring="accuracy", cv=10, verbose=1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

y_pred = grid.best_estimator_.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))
print(pd.DataFrame(grid.best_estimator_.feature_importances_, X.columns).sort_values(0, ascending=False))
print(grid.best_estimator_.feature_importances_.sum())