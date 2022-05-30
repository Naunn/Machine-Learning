# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:06:25 2022

@author: Bartosz Lewandowski
"""
# %% import
# data
from data import df
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split

# general
import numpy as np
import pandas as pd

# visualisation
import plotly.express as px
import plotly.io as pio #Niezbędne do wywoływania interaktywnych rysunków
#pio.renderers.default = 'svg' #Wykresy w Spyder (statyczne)
pio.renderers.default = 'browser' #Wykresy w przeglądarce (interaktywne)

# models
from statsmodels.api import OLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from keras.layers import Dense, Activation
from keras.models import Sequential
# %% Exploration
# =============================================================================
# 1. Understand the problem. We'll look at each variable and do a philosophical 
# analysis about their meaning and importance for this problem.
# =============================================================================
# Prawdopodobnie jest to jakies fizyczne doswiadczenie, wahadla lub czegos z ruchem (sinusoidy).
# =============================================================================

# 2. Univariable study. We'll just focus on the dependent variable ('Target') 
# and try to know a little bit more about it.
# =============================================================================
# Jest to zapewne jakas miara odleglosci pokonanej przez obiekt, czy pchnieciu.
# =============================================================================

# 3. Multivariate study. We'll try to understand how the dependent variable 
# and independent variables relate.
# =============================================================================
# Najprawdopodobniej sa to zmienne srodowiskowe takie jak wiatr, temp, wilgotnosc,
# powloka, waga kulki, dlugosc linki (na ktorej dynda kulka) itp.
# Relacja miedzy nimi jest taka, ze mamy wszelkie informacje jakie towarzyszyly
# pchnieciu kulki, poza tym, z jaka sila zostala pchnieta.
# Dodatkowo, firma ma wplyw (wiec za pewnie manipuluje w sposob kontrolowany) na zmienne
# Var_mass i Var_LT.
# =============================================================================

# 4. Basic cleaning. We'll clean the dataset and handle the missing data, outliers 
# and categorical variables.
# =============================================================================
# Takich rzeczy, w doswiadczeniu fizycznym (z duzym prawdopodobienstwem) nie uswiadczymy.
# =============================================================================

# 5. Test assumptions. We'll check if our data meets the assumptions required 
# by most multivariate techniques.
# =============================================================================
# Tutaj ... zobaczymy co z tego bedzie.
# =============================================================================

# Based on Hair et al. (2013), chapter 'Examining your data'.
# =============================================================================

desc = df.describe()

df.Target.describe()
# count    399.000000
# mean      14.678947
# std        2.574374
# min        4.472222
# 25%       13.222222
# 50%       15.500000
# 75%       15.694444
# max       26.861111
# Name: Target, dtype: float64

cel = df['Target'] # Wygląd blisko rozkładu normalnego
#skewness and kurtosis
print("Skewness: %f" % cel.skew())
# lekko pozytywna skosnosc => prawostronna asymetria => Dominanta < Mediana < Średnia
print("Kurtosis: %f" % cel.kurt())
# kurtoza dodatnia i realtywnie duza, wiec wartosci skupione wokol sredniej

# Uwaga! Jest to najpewniej doswiadczenie fizyczne,
# gdzie Var_mass i Var_LT sa zmieniane przez firme.

# Poszukiwanie pewnych grup w zmiennych manipulowanych przez firme.
fig = px.scatter(df, x = 'Var_LT', y = 'Var_mass')
# fig.show()

# Wszelkie dane (poza Var_mass i Var_LT) sa z przedzialu (0,1)
df.Var_av.describe()

# =============================================================================
# AIC (Akaike information criterion) = 2*k - 2*ln(L),
# where k is number of parameters and L is likelihood function.
# =============================================================================
# Interpretation: I want as few parameters as possible (prevent overfitting)
# with as high likelihood function value as possible.
# "Lower AIC via higher log likelihood or less parameters".
# =============================================================================

# =============================================================================
# BIC (Bayesian information criterion) = k*ln(n)-2*ln(L),
# where n is number of samples, k is number of parameters and L is likelihood function.
# =============================================================================
# Interpretation: Similar to AIK but also take the number of sample used
# in training into account. We will use BIC when AIC is the same for models,
# but we use different number of samples to training both.
# "Lower BIC via higher log likelihood or less parameters or less samples used in fitting".
# =============================================================================

# Note. In our case, BIC is not essential. Also, it is important
# to compare models with variables on the same scales. For example, the
# AIC score will be different for original data and normalized data.

# %% Visualization
fig = px.line(df, x=[_ for _ in range(1,len(df)+1)], y = list(df.columns))
fig.show()
corr_matrix = df.corr() # Correlation Matrix
# %% Data split
# In this case, We will split data rather then use cross-validation.

# Set aside 20% of train and test data for evaluation.
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],
                                                    df.Target,
                                                    test_size=0.2,
                                                    shuffle = True,
                                                    random_state = 8)

# Use the same function above for the validation set.
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  shuffle = True,
                                                  random_state= 8)

print("X_train shape: {}".format(X_train.shape))
print("X_val shape: {}".format(X_val.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y val shape: {}".format(y_test.shape))
print("y_test shape: {}".format(y_test.shape))
# %% scaling

# (W przypadku roznych skal)
# Standardization is good to use when our data follows a normal distribution.
# It can be used in a machine learning algorithm where we make assumptions about
# the distribution of data like linear regression etc
scaler = StandardScaler()
X_data = df.iloc[:,0:10]
scaled_data = pd.DataFrame(scaler.fit_transform(X_data), columns=X_data.columns) 
scaled_data

# (W przypadku niejednolitego pomiaru)
# Normalization is preferred over standardization when our data doesn’t
# follow a normal distribution. It can be useful in those machine learning
# algorithms that do not assume any distribution of data like the k-nearest
# neighbor and neural networks.
min_max_scaler = MinMaxScaler()
min_max_data = df[['Var_LT','Var_mass','Target']]
min_max_scalled_data = pd.DataFrame(min_max_scaler.fit_transform(min_max_data),
                                    columns=min_max_data.columns)
normalized_data = df.copy()
normalized_data[['Var_LT',
             'Var_mass',
             'Target']] = min_max_scalled_data[['Var_LT',
                                                'Var_mass',
                                                'Target']]

desc = normalized_data.describe()
                                                
# fig = px.line(scaled_data,
#               x=[_ for _ in range(1,len(df)+1)],
#               y = list(scaled_data.columns))
# fig.show()
# %% varaible check with lm
# Standarized
data = scaled_data.copy()
ols = OLS(data.Target, data.iloc[:,0:9]).fit()
print(ols.summary())

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                 Target   R-squared (uncentered):                   0.212
# Model:                            OLS   Adj. R-squared (uncentered):              0.196
# Method:                 Least Squares   F-statistic:                              13.18
# Date:                Wed, 18 May 2022   Prob (F-statistic):                    6.91e-17
# Time:                        22:35:33   Log-Likelihood:                         -518.52
# No. Observations:                 399   AIC:                                      1053.
# Df Residuals:                     391   BIC:                                      1085.
# Df Model:                           8                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Var_av         0.0295      0.013      2.315      0.021       0.004       0.055
# Var_LT         0.2354      0.048      4.921      0.000       0.141       0.329
# Var_mass       0.0276      0.045      0.611      0.542      -0.061       0.116
# Var1           0.0234      0.124      0.189      0.850      -0.220       0.267
# Var2           0.4996      0.168      2.977      0.003       0.170       0.830
# Var3          -0.3983      0.129     -3.093      0.002      -0.651      -0.145
# Var4           0.1679      0.153      1.094      0.275      -0.134       0.470
# Var5          -0.3206      0.157     -2.046      0.041      -0.629      -0.012
# Var6           0.3352      0.113      2.969      0.003       0.113       0.557
# ==============================================================================
# Omnibus:                       88.479   Durbin-Watson:                   1.474
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              718.263
# Skew:                           0.676   Prob(JB):                    1.07e-156
# Kurtosis:                       9.432   Cond. No.                     7.63e+15
# ==============================================================================
# So we will continue with Var_mass, Var1, Var4 and (maybe) Var_av

# Normalized
data = normalized_data.copy()
ols = OLS(data.Target, data.iloc[:,0:9]).fit()
print(ols.summary())

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                 Target   R-squared (uncentered):                   0.941
# Model:                            OLS   Adj. R-squared (uncentered):              0.939
# Method:                 Least Squares   F-statistic:                              773.7
# Date:                Wed, 18 May 2022   Prob (F-statistic):                   2.16e-234
# Time:                        22:36:35   Log-Likelihood:                          298.20
# No. Observations:                 399   AIC:                                     -580.4
# Df Residuals:                     391   BIC:                                     -548.5
# Df Model:                           8                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Var_av         0.0637      0.010      6.692      0.000       0.045       0.082
# Var_LT         0.3635      0.030     12.159      0.000       0.305       0.422
# Var_mass       0.0941      0.020      4.773      0.000       0.055       0.133
# Var1          -0.0577      0.088     -0.656      0.512      -0.230       0.115
# Var2           0.4814      0.138      3.494      0.001       0.211       0.752
# Var3          -0.2239      0.107     -2.100      0.036      -0.433      -0.014
# Var4           0.0662      0.125      0.530      0.597      -0.180       0.312
# Var5          -0.1047      0.125     -0.837      0.403      -0.351       0.141
# Var6           0.2564      0.088      2.911      0.004       0.083       0.430
# ==============================================================================
# Omnibus:                       52.443   Durbin-Watson:                   1.513
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              215.206
# Skew:                           0.480   Prob(JB):                     1.86e-47
# Kurtosis:                       6.467   Cond. No.                     8.62e+15
# ==============================================================================

prstd, iv_l, iv_u = wls_prediction_std(ols)

df_ols = pd.DataFrame({'Target': data.Target,
                       'y_hat': ols.fittedvalues,
                       'iv_l': iv_l,
                       'iv_u': iv_u})

fig = px.line(df_ols,
              x=[_ for _ in range(1,len(df_ols)+1)],
              y = ['Target','y_hat','iv_l','iv_u'])
# fig.show()

# %% Box-Cox data
df_Box_Cox = df.copy()

df_Box_Cox = df_Box_Cox.apply(lambda x: boxcox(x)[0])

fig = px.line(df_Box_Cox,
              x=[_ for _ in range(1,len(df_Box_Cox)+1)],
              y = df_Box_Cox.columns)
# fig.show()

df_Box_Cox_norm = normalized_data.copy()
df_Box_Cox_norm = df_Box_Cox_norm.apply(lambda x: x + 0.000001) # trik na problem z wystepowaniem zer
df_Box_Cox_norm = df_Box_Cox_norm.apply(lambda x: boxcox(x)[0])

fig = px.line(df_Box_Cox_norm,
              x=[_ for _ in range(1,len(df_Box_Cox_norm)+1)],
              y = df_Box_Cox_norm.columns)
# fig.show()

# %% ANN regression

# =============================================================================
# Nh = Ns/(alpha∗ (Ni + No))
# Ni = number of input neurons.
# No = number of output neurons.
# Ns = number of samples in training data set.
# alpha = an arbitrary scaling factor usually 2-10.
# =============================================================================
# In our case we have: Nh = 399/(alpha*(399+1)) => 399/(4*(32+1)) ~ 2

# Initialising the ANN
# model = Sequential()

# # Adding the input layer and the first hidden layer
# model.add(Dense(32, activation = 'relu', input_dim = 9))

# # Adding the second hidden layer
# model.add(Dense(units = 32, activation = 'relu'))

# # Adding the third hidden layer
# model.add(Dense(units = 32, activation = 'relu'))

# # Adding the output layer
# model.add(Dense(units = 1))

# model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics=['mse', 'mae', 'mape'])

# X = np.array(normalized_data.iloc[:,:-1])
# Y = np.array(normalized_data.iloc[:,-1])

# fitted_model = model.fit(X, Y, epochs=100, batch_size=10, verbose=2)
# fitted_model.history

# y_pred = model.predict(X) # "sztuczna" predykcja

# from LinearRegressionMatrixImplementation import T

# tst = pd.DataFrame({'Target':normalized_data.Target,
#                     'pred':T(y_pred)[0]})

# fig = px.line(tst,
#               x=[_ for _ in range(1,len(y_pred)+1)],
#               y = tst.columns)
# # fig.show()