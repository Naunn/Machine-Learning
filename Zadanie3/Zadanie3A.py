import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

Var_av = pd.read_csv("Zadanie_regresja\Var_av.txt", header = None, names = ["Var_av"])
Var_LT = pd.read_csv("Zadanie_regresja\Var_LT.txt", header = None, names = ["Var_LT"])
Var_mass = pd.read_csv("Zadanie_regresja\Var_mass.txt", header = None, names = ["Var_mass"])
Var1 = pd.read_csv("Zadanie_regresja\Var1.txt", header = None, names = ["Var1"])
Var2 = pd.read_csv("Zadanie_regresja\Var2.txt", header = None, names = ["Var2"])
Var3 = pd.read_csv("Zadanie_regresja\Var3.txt", header = None, names = ["Var3"])
Var4 = pd.read_csv("Zadanie_regresja\Var4.txt", header = None, names = ["Var4"])
Var5 = pd.read_csv("Zadanie_regresja\Var5.txt", header = None, names = ["Var5"])
Var6 = pd.read_csv("Zadanie_regresja\Var6.txt", header = None, names = ["Var6"])
Target = pd.read_csv("Zadanie_regresja\Target.txt", header = None, names = ["Target"])
data = pd.concat([Var_av, Var_LT, Var_mass, Var1, Var2, Var3, Var4, Var5, Var6, Target], axis = 1)

inputs = data.iloc[:, 0:8].values
outputs = data.iloc[:, 9].values

X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regresor = ElasticNet(alpha = 0.2, l1_ratio = 0)
regresor.fit(X_train, Y_train)

predicted = regresor.predict(X_test)

coefficients = regresor.coef_
print('Coefficients: \n', coefficients)

error = np.mean((predicted - Y_test) ** 2)
error = np.sqrt(error)
print('Standard deviation of residuals: ', error)

mean = np.mean(Y_test)
error2l = np.sum((predicted - mean) ** 2)
error2m = np.sum((Y_test - mean) ** 2)
error2 = 1 - (error2l / error2m)
print('Convergence coefficient: ', error2)