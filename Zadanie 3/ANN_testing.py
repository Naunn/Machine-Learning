# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:02:36 2022

@author: Bartosz Lewandowski
"""

# %% Import
# data
from data import normalized_data
from sklearn.model_selection import train_test_split

# general
import pandas as pd
import tensorflow as tf

# visualisation
import plotly.express as px
import plotly.io as pio #Niezbędne do wywoływania interaktywnych rysunków
#pio.renderers.default = 'svg' #Wykresy w Spyder (statyczne)
pio.renderers.default = 'browser' #Wykresy w przeglądarce (interaktywne)

# models
from LinearRegressionMatrixImplementation import SSE,SSR,SST,R_Squared,R_Squared_Adj,calculate_aic, T
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# %% Preparing splitted Data
data = normalized_data.copy()

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],
                                                    data.Target,
                                                    test_size=0.2,
                                                    shuffle = True,
                                                    random_state = 8)
# Adding additional validation set
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  shuffle = True,
                                                  random_state= 8)

# %% Testing

# =============================================================================
# selection of the optimizer
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/MomentumOptimizer
# To switch to native TF2 style, please directly use
# tf.keras.optimizers.SGD with the momentum argument.
# https://www.quora.com/Which-optimizer-in-TensorFlow-is-best-suited-for-learning-regression4
# 
# "Adam", "SGD", tf.keras.optimizers.SGD(momentum = 0.1)
# =============================================================================

# =============================================================================
# selection of the activation function
# https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
# 
# "linear", "relu"
# =============================================================================

def testing_ANN(activation_fun: str,
                optimizer,
                batch_size: int,
                train_data,
                train_target,
                val_data,
                val_target):
    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation = activation_fun, input_dim = 9))

    # Adding the second hidden layer
    model.add(Dense(units = 32, activation = activation_fun))

    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = activation_fun))

    # Adding the output layer
    model.add(Dense(units = 1))
    
    model.compile(optimizer = optimizer,
                  loss = 'mean_squared_error',
                  metrics=['mse', 'mae', 'mape'])
    
    # Verbose is visual setting.
    # With batch_size: 25, 50, there is a problem with missing best fitting, so batch_size = 10
    model.fit(train_data, train_target, epochs=100, batch_size=batch_size, verbose=2)
    
    pred = model.predict(val_data.iloc[:,:9])

    val_data['Target'] = val_target
    val_data['pred'] = T(pred)[0]
    val_data['eps'] = val_data.apply(lambda x: x['pred']-x['Target'],axis=1)
    
    mae = mean_absolute_error(val_data['Target'],val_data['pred'])
    mape = mean_absolute_percentage_error(val_data['Target'],val_data['pred'])
    mse = mean_squared_error(val_data['Target'],val_data['pred'])
    aic = calculate_aic(len(val_data), mse, 9+1)
    
    return activation_fun, optimizer, batch_size, mae, mape, mse, aic

testing_df = pd.DataFrame(columns=["Activation function",
                                   "Optimizer",
                                   "Batch size",
                                   "MAE",
                                   "MAPE",
                                   "MSE",
                                   "AIC"])

SGD_momentum = tf.keras.optimizers.SGD(momentum = 0.5)
optimators = ["Adam", "SGD", SGD_momentum]
activation_functions = ["linear", "relu"]
batch = [1, 5, 10, 15, 20]

for opt in optimators:
    for act_fun in activation_functions:
        for b in batch:
            values = testing_ANN(act_fun,
                                 opt,
                                 b,
                                 X_train,
                                 y_train,
                                 X_val,
                                 y_val)
        
            testing_df = testing_df.append({"Activation function": values[0],
                              "Optimizer": values[1],
                              "Batch size": values[2],
                              "MAE": values[3],
                              "MAPE": values[4],
                              "MSE": values[5],
                              "AIC": values[6]},
                              ignore_index=True)

# %% best model
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
model = Sequential()

model.add(Dense(32, activation = "relu", input_dim = 9))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 1))
    
model.compile(optimizer = 'Adam',
              loss = 'mean_squared_error',
              metrics=['mse', 'mae', 'mape'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# Training will stop when the chosen performance measure stops improving.
# To discover the training epoch on which training was stopped, the “verbose”
# argument can be set to 1. Once stopped, the callback will print the epoch number.

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val.iloc[:,:9], y_val),
                    epochs=250,
                    batch_size=5,
                    verbose=2)
                    # callbacks = [es])

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
# the value when some element of the y_true is zero is arbitrarily high because
# of the division by epsilon
#mean_absolute_percentage_error(X_test['Target'],X_test['pred'])
mape = history.history['mape'][-1]
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