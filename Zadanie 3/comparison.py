# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:57:51 2022

@author: Bartosz Lewandowski
"""
# %% Import
# data
from data import df
from data_mining import scaled_data, normalized_data, df_Box_Cox, df_Box_Cox_norm
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
from LinearRegressionMatrixImplementation import MultipleRegression,SSE,SSR,SST,R_Squared,R_Squared_Adj,calculate_aic, T   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from statsmodels.tsa.stattools import acf
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.api import OLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from keras.layers import Dense, Activation
from keras.models import Sequential

# %% Preparing splitted Data
data = normalized_data.copy()
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],
                                                    data.Target,
                                                    test_size=0.2,
                                                    shuffle = True,
                                                    random_state = 8)

# %% Linear multiple regression
cols = [0,1,2,3,4,5,6,7,8]
XTX, coeff = MultipleRegression(X_train, cols, np.array(y_train))
print("coeff:",coeff)

X_test['Target'] = y_test
X_test['pred'] = X_test.apply(lambda x: coeff[0][0]
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

X_test['eps'] = X_test.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(X_test['eps']))
ssr = SSR(list(X_test['Target']),list(X_test['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(X_test))
mae = mean_absolute_error(X_test['Target'],X_test['pred'])
mape = mean_absolute_percentage_error(X_test['Target']+0.001,X_test['pred'])
mse = mean_squared_error(X_test['Target'],X_test['pred'])
aic = calculate_aic(len(X_test), mse, len(cols)+1)


print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("MAE:         ",mae)
print("MAPE:        ",mape)
print("MSE:         ",mse)
print("AIC:         ",aic)

fig = px.line(X_test, x=[_ for _ in range(1,len(X_test)+1)], y = ["Target","pred"])
fig.show()

# %% LinearRegression from sklearn all
reg = LinearRegression().fit(X_train, y_train)
print("coeff:",reg.intercept_, reg.coef_)

X_test['Target'] = y_test
X_test['pred'] = X_test.apply(lambda x: reg.intercept_
                      +x['Var_av']*reg.coef_[0]
                      +x['Var_LT']*reg.coef_[1]
                      +x['Var_mass']*reg.coef_[2]
                      +x['Var1']*reg.coef_[3]
                      +x['Var2']*reg.coef_[4]
                      +x['Var3']*reg.coef_[5]
                      +x['Var4']*reg.coef_[6]
                      +x['Var5']*reg.coef_[7]
                      +x['Var6']*reg.coef_[7]
                      ,axis=1)

X_test['eps'] = X_test.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(X_test['eps']))
ssr = SSR(list(X_test['Target']),list(X_test['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(X_test))
mae = mean_absolute_error(X_test['Target'],X_test['pred'])
mape = mean_absolute_percentage_error(X_test['Target']+0.001,X_test['pred'])
mse = mean_squared_error(X_test['Target'],X_test['pred'])
aic = calculate_aic(len(X_test), mse, len(cols)+1)


print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("MAE:         ",mae)
print("MAPE:        ",mape)
print("MSE:         ",mse)
print("AIC:         ",aic)

fig = px.line(X_test, x=[_ for _ in range(1,len(X_test)+1)], y = ["Target","pred"])
fig.show()

# %% OLS from scipy.stats
ols = OLS(y_train, X_train).fit()
print(ols.summary())

X_test['Target'] = y_test
X_test['pred'] = ols.predict(X_test.iloc[:,:9])
X_test['eps'] = X_test.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(X_test['eps']))
ssr = SSR(list(X_test['Target']),list(X_test['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, len(cols), len(X_test))
mae = mean_absolute_error(X_test['Target'],X_test['pred'])
mape = mean_absolute_percentage_error(X_test['Target']+0.001,X_test['pred'])
mse = mean_squared_error(X_test['Target'],X_test['pred'])
aic = calculate_aic(len(X_test), mse, len(cols)+1)


print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("MAE:         ",mae)
print("MAPE:        ",mape)
print("MSE:         ",mse)
print("AIC:         ",aic)

fig = px.line(X_test, x=[_ for _ in range(1,len(X_test)+1)], y = ["Target","pred"])
fig.show()
# =============================================================================
# # model fitting
# prstd, iv_l, iv_u = wls_prediction_std(ols)
# 
# df_ols = pd.DataFrame({'Target': y_train,
#                        'y_hat': ols.fittedvalues,
#                        'iv_l': iv_l,
#                        'iv_u': iv_u})
# 
# fig = px.line(df_ols,
#               x=[_ for _ in range(1,len(df_ols)+1)],
#               y = ['Target','y_hat','iv_l','iv_u'])
# fig.show()
# =============================================================================

# %% ANN regression

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

# Verbose is visual setting.
# With batch_size: 25, 50, there is a problem with missing best fitting.
# We go with batch_size = 10 and many (min. 1000) epochs.
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)

pred = model.predict(X_test.iloc[:,:9])

X_test['Target'] = y_test
X_test['pred'] = T(pred)[0]
X_test['eps'] = X_test.apply(lambda x: x['pred']-x['Target'],axis=1)

sse = SSE(list(X_test['eps']))
ssr = SSR(list(X_test['Target']),list(X_test['pred']))
sst = SST(sse, ssr)
r_squared = R_Squared(ssr, sst)
r_squared_adj = R_Squared_Adj(sse, sst, 9, len(X_test))
mae = mean_absolute_error(X_test['Target'],X_test['pred'])
mape = history.history['mape'][-1]
# mape = mean_absolute_percentage_error(X_test['Target'],X_test['pred'])
mse = mean_squared_error(X_test['Target'],X_test['pred'])
aic = calculate_aic(len(X_test), mse, 9+1)


print("SSE:         ",sse)
print("SSR:         ",ssr)
print("SST:         ",sst)
print("R^2:         ",r_squared)
print("R^2 adjusted:",r_squared_adj)
print("MAE:         ",mae)
print("MAPE:        ",mape)
print("MSE:         ",mse)
print("AIC:         ",aic)

fig = px.line(X_test, x=[_ for _ in range(1,len(X_test)+1)], y = ["Target","pred"])
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









