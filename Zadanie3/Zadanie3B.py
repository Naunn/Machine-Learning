import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Telephones.csv')

inputs = dataset.iloc[:, 0:19].values
outputs = dataset.iloc[:, 20].values

X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, min_samples_split = 5, random_state = 0)
classifier.fit(X_train, Y_train)

predicted = classifier.predict(X_test)

print(confusion_matrix(Y_test, predicted))
print(accuracy_score(Y_test, predicted))
