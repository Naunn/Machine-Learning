# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:57:51 2022

@author: Bartosz Lewandowski
"""
# %% Import
# data
from data import df
from data_mining import scaled_data, normalized_data, df_Box_Cox, df_Box_Cox_norm

# general
import numpy as np
import pandas as pd

# visualisation
import plotly.express as px
import plotly.io as pio #Niezbędne do wywoływania interaktywnych rysunków
#pio.renderers.default = 'svg' #Wykresy w Spyder (statyczne)
pio.renderers.default = 'browser' #Wyrkesy w przeglądarce (interaktywne)

# models
from LinearRegressionMatrixImplementation import MultipleRegression,SSE,SSR,SST,R_Squared,R_Squared_Adj,calculate_aic, T   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.stattools import acf
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.api import OLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from keras.layers import Dense, Activation
from keras.models import Sequential
# %% All columns
data = df.copy()
print(data.head())

cols = [0,1,2,3,4,5,6,7,8]
XTX, coeff = MultipleRegression(data, cols, 9)
print("coeff:",coeff)

data['pred'] = data.apply(lambda x: coeff[0][0]
                      +x['Var_av']*coeff[1][0]
                      +x['Var_LT']*coeff[2][0]
                      +x['Var_mass']*coeff[3][0]
                      +x['Var1']*coeff[4][0]
                      +x['Var2']*coeff[5][0]
                      +x['Var3']*coeff[6][0]
                      +x['Var4']*coeff[7][0]
                      +x['Var5']*coeff[8][0]
                      +x['Var6']*coeff[9][0]
                      ,axis=1)

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(data))
aic = calculate_aic(len(data), mean_squared_error(data['Target'],data['pred']), len(cols)+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data, x=[_ for _ in range(1,len(data)+1)], y = ["Target","pred"])
# fig.show()
# %% Var_LT and Var_mass
data = df.copy()
print(data.head())

cols = [1,2]
XTX, coeff = MultipleRegression(data, cols, 9)
print("coeff:",coeff)

data['pred'] = data.apply(lambda x: coeff[0][0]
                      +x['Var_LT']*coeff[1][0]
                      +x['Var_mass']*coeff[2][0]
                      ,axis=1)

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(data))
aic = calculate_aic(len(data), mean_squared_error(data['Target'],data['pred']), len(cols)+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data, x=[_ for _ in range(1,len(data)+1)], y = ["Target","pred"])
# fig.show()
# %% Var_mass, Var1, Var4 standarized
data = scaled_data.copy()
print(data.head())

cols = [2,3,6]
XTX, coeff = MultipleRegression(data, cols, 9)
print("coeff:",coeff)

data['pred'] = data.apply(lambda x: coeff[0][0]
                      +x['Var_mass']*coeff[1][0]
                      +x['Var1']*coeff[2][0]                      
                      +x['Var4']*coeff[3][0]
                      ,axis=1)

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(data))
aic = calculate_aic(len(data), mean_squared_error(data['Target'],data['pred']), len(cols)+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data, x=[_ for _ in range(1,len(data)+1)], y = ["Target","pred"])
# fig.show()
# %% Var1, Var4, Var5 normilized
data = normalized_data.copy()
print(data.head())

cols = [3,6,7]
XTX, coeff = MultipleRegression(data, cols, 9)
print("coeff:",coeff)

data['pred'] = data.apply(lambda x: coeff[0][0]
                      +x['Var1']*coeff[1][0]
                      +x['Var4']*coeff[2][0]                      
                      +x['Var5']*coeff[3][0]
                      ,axis=1)

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(data))
aic = calculate_aic(len(data), mean_squared_error(data['Target'],data['pred']), len(cols)+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data, x=[_ for _ in range(1,len(data)+1)], y = ["Target","pred"])
# fig.show()
# %% LinearRegression from sklearn all
data = normalized_data.copy()

X = np.array(data.iloc[:,0:8])
y = np.array(data.iloc[:,9])

reg = LinearRegression().fit(X, y)
print("coeff:",reg.intercept_, reg.coef_)
print("R^2:",reg.score(X, y))

data['pred'] = data.apply(lambda x: reg.intercept_
                      +x['Var_av']*reg.coef_[0]
                      +x['Var_LT']*reg.coef_[1]
                      +x['Var_mass']*reg.coef_[2]
                      +x['Var1']*reg.coef_[3]
                      +x['Var2']*reg.coef_[4]
                      +x['Var3']*reg.coef_[5]
                      +x['Var4']*reg.coef_[6]
                      +x['Var5']*reg.coef_[7]
                      ,axis=1)

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(data))
aic = calculate_aic(len(data), mean_squared_error(data['Target'],data['pred']), len(cols)+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data, x=[_ for _ in range(1,len(data)+1)], y = ["Target","pred"])
# fig.show()
# %% OLS from scipy.stats
data = df_Box_Cox.copy()
ols = OLS(data.Target, data.iloc[:,0:9]).fit()
print(ols.summary())

prstd, iv_l, iv_u = wls_prediction_std(ols)

df_ols = pd.DataFrame({'Target': data.Target,
                       'y_hat': ols.fittedvalues,
                       'iv_l': iv_l,
                       'iv_u': iv_u})

fig = px.line(df_ols,
              x=[_ for _ in range(1,len(df_ols)+1)],
              y = ['Target','y_hat','iv_l','iv_u'])
fig.show()
# %% ANN regression
data = normalized_data.copy()
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 9))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics=['mse', 'mae', 'mape'])

X = np.array(data.iloc[:,:-1])
Y = np.array(data.iloc[:,-1])

# Verbose is visual setting.
# With batch_size: 25, 50, there is a problem with missing best fitting.
# We go with batch_size = 10 and many (min. 1000) epochs.
fitted_model = model.fit(X, Y, epochs=1000, batch_size=10, verbose=2)

y_pred = model.predict(X) # "sztuczna" predykcja

data['pred'] = T(y_pred)[0]

data['eps'] = data.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(data['eps']))
ssr = SSR(list(data['Target']),list(data['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, 9, len(data))
aic = calculate_aic(len(data), fitted_model.history['mse'][-1], 9+1)

print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("AIC:         ",aic)

fig = px.line(data,
              x=[_ for _ in range(1,len(y_pred)+1)],
              y = data.columns)
fig.show()
# %% prophet (nietrafiony pomysl)
# =============================================================================
# data = df.copy()
# acf_Target = acf(data.Target) # Target is i.i.d. so we can try to use time series
# 
# data = {'ds': pd.date_range("2019-01-01", periods=385),
#       'y': data['Target'][:385],
#       'Var_mass': data['Var_mass'][:385],
#       'Var_LT': data['Var_LT'][:385]}
# data_prophet = pd.DataFrame(data)
# print(data_prophet)
# 
# m = Prophet()
# m.add_regressor('Var_mass')
# m.add_regressor('Var_LT')
# m.fit(data_prophet)
# 
# future = m.make_future_dataframe(periods=14)
# future['Var_mass'] = data['Var_mass']
# future['Var_LT'] = data['Var_LT']
# print(future.tail(14))
# 
# forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# 
# fig1 = m.plot(forecast)
# 
# df_cv = cross_validation(m, horizon=56)
# df_p = performance_metrics(df_cv)
# print(df_p.head())
# #                     horizon        mse       rmse  ...     mdape     smape  coverage
# # 0 0 days 00:00:00.000000056  114.77731  10.713417  ...  0.092273  0.138056  0.848806
# =============================================================================









