import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('../../data/tweets.csv')
# print(df['is_viral'].value_counts())
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

viral_predict = clf.predict([[70, 5, 1], [130, 50, 1]])
# print(viral_predict)

y_pred = clf.predict(x_test)
print(y_pred)
# print(y_test)

print(accuracy_score(y_test, y_pred))