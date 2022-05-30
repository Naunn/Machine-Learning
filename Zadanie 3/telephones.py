# -*- coding: utf-8 -*-
"""
Created on Fri May 27 22:17:35 2022

@author: Bartosz Lewandowski
"""
# %% Packages
# data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# general
import pandas as pd
import numpy as np

# visualisation
import plotly.express as px
import plotly.io as pio #Niezbędne do wywoływania interaktywnych rysunków
#pio.renderers.default = 'svg' #Wykresy w Spyder (statyczne)
pio.renderers.default = 'browser' #Wykresy w przeglądarce (interaktywne)

# models
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix,classification_report
# %% Data
telephones_df = pd.read_csv(r'Telephones.csv')
telephones_df.columns
# %% Exploration
# =============================================================================
# 1. Understand the problem. We'll look at each variable and do a philosophical 
# analysis about their meaning and importance for this problem.
# =============================================================================
# Naszym celem jest klasyfikatora, który będzie przyzielał odpowiedni półki cenowe
# price_range: Półka cenowa; (CEL)
# battery_power: Pojemność baterii w mAh;
# blue: wewnętrzny filtr światła niebieskiego;
# clock_speed: częstotliwość/prędkosć procesora;
# dual_sim: Czy telefon wspiera więcej niż jedną kartę SIM;
# fc: Rozdzielczość aparatu z przodu w megapikselach;
# four_g: Czy posiada zdolność do obsługi sieci 4G;
# int_memory: Pamięć wbudowana w Gb;
# m_dep: Grubosć telefonu w cm;
# mobile_wt: Waga telefonu;
# n_core: liczba rdzeni procesora;
# pc: Liczba pikseli przedniej kamery;
# px_height: Rodzielczosć telefonu (wysokosć);
# px_width: Rodzielczosć telefonu (szerokosć);
# ram: Pamięć RAM w MB;
# sc_h: Wysokosć ekranu;
# sc_w: Szerokosć ekranu;
# talk_time: najdłuższy czas pracy na jednym ładowaniu akumulatora, gdy użytkownik rozmawia;
# three_g: Czy posiada 3G;
# touch_screen: Czy posiada ekran dotykowy;
# wifi: Czy posiada WiFi

desc = telephones_df.describe()

cel = telephones_df['price_range'] # Wygląd blisko rozkładu normalnego
#skewness and kurtosis
print("Skewness: %f" % cel.skew())
# brak skosnosci => rozklad symetryczny
print("Kurtosis: %f" % cel.kurt())
# kurtoza ujemna, wiec wartosci rozproszone (od sredniej)

# =============================================================================

# 2. Univariable study. We'll just focus on the dependent variable ('Target') 
# and try to know a little bit more about it.
# =============================================================================
# Zmieniajac wartosc "price_range" mozemy podejrzec zachowanie cech
desc = telephones_df[telephones_df.price_range == 0].describe()

# =============================================================================

# 3. Multivariate study. We'll try to understand how the dependent variable 
# and independent variables relate.
# =============================================================================
# No need to standardize. Because by definition the correlation coefficient is
# independent of change of origin and scale. As such standardization
# will not alter the value of correlation.
corr = telephones_df.corr()
# od razu widac, ze RAM jest niezwykle istotny przy klasyfikacji
# dodatkowo mozna dorzucic: battery_power, px_height, px_width

# =============================================================================

# 4. Basic cleaning. We'll clean the dataset and handle the missing data, outliers 
# and categorical variables.
# =============================================================================
# Nalezy dodac +1 do wszyskich wartosci. Powinno to zapobiec przyszlym problemom
# z klasyfikowaniem lub eksploracja danych.
telephones_df['price_range'] = telephones_df['price_range']+1

# Podejrzany sc_w (min wartosc = 0)
tst = telephones_df[telephones_df.sc_w == 0] #180 pozycji
telephones_df[telephones_df.sc_w != 0].sc_w.describe()

# int(round(np.random.normal(6.337363, 4.152062, 1)[0],0))

telephones_df['sc_w'] = telephones_df[(telephones_df.sc_w == 0) |
                                      (telephones_df.sc_w < 0) |
                                      (telephones_df.sc_h <= telephones_df.sc_w)].sc_w.apply(
                                          lambda x: x + int(round(np.random.normal(6.337363,
                                                                                   4.152062,
                                                                                   1)[0],0)))
telephones_df[telephones_df.sc_w == 0]
telephones_df[telephones_df.sc_w < 0]
telephones_df.corr() # Koniecznie sprawdzić, czy nie wpłyneło na macierz korelacji
# =============================================================================

# 5. Test assumptions. We'll check if our data meets the assumptions required 
# by most multivariate techniques.
# =============================================================================
# Glowne pytanie, co chcemy maksymalizowac?
# Accuracy - gdy mamy zbalansowany zbiór testowy;
# Precision - w sytuacji, gdy false positive sa bardziej
# niebezpieczne niz false negative (np. testy ciazowe);
# Recall - odwrotnie niz dla precision, szczególnie wazne w
# medycynie (np. diagnostyka nowotworów);
# F1 - gdy false positive i negative sa tak samo kosztowne i gdy
# duzo negatywów jest klasyfikowanych poprawnie.

# W naszym wypadku, skupimy sie na "accuaracy" glownie z uwagi na to, ze 
# rownie niekorzystne moze sie okazac zawyzenie jak i zanizenie polki cenowej.
# =============================================================================

# Based on Hair et al. (2013), chapter 'Examining your data'.
# =============================================================================

# %% Data visualization
# fig = px.scatter(telephones_df,
#               x=[_ for _ in range(1,len(telephones_df)+1)],
#               y = list(telephones_df.columns))
# fig.show()
# Bagno widac ...

fig = px.scatter(telephones_df,
              x= 'ram',
              y = 'battery_power',
              color = 'price_range')
fig.show()
# Obiecujący wynik

fig = px.scatter(telephones_df,
              x= 'ram',
              y = 'px_width',
              color = 'price_range')
fig.show()
# Obiecujący wynik

fig = px.scatter(telephones_df,
              x= 'ram',
              y = 'int_memory',
              color = 'price_range')
fig.show()
# Obiecujący wynik
# Ewidentnie "RAM" swietnie oddaje dopasowanie do kategorii cenowej.

# %% Data preparation
# min_max_scaler = MinMaxScaler()
# min_max_data = telephones_df[['battery_power','px_height','px_width','ram']]
# min_max_scalled_data = pd.DataFrame(min_max_scaler.fit_transform(min_max_data),
#                                     columns=min_max_data.columns)
# normalized_data = min_max_scalled_data.join(telephones_df.price_range)

data = telephones_df[['battery_power','px_height','px_width','ram','price_range']].copy()

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],
                                                    data.price_range,
                                                    test_size=0.2,
                                                    shuffle = True,
                                                    random_state = 8)
# Adding additional validation set
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  shuffle = True,
                                                  random_state= 8)
# %% SVM

# Bedziemy glownie minimalizowac recall dla 4 i 3 kategorii, z uwagi na to, ze ich
# zanizenie moze byc bardziej kosztowne, przy tym miejac accuaracy na uwadze.

# Classic method
# clf = svm.SVC(decision_function_shape="ovo") # "one-versus-one"
# clf.fit(normalized_data.iloc[:,:4], normalized_data.price_range)
# clf.get_params()

# Method using pipeline

# kernels = {’rbf’,‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’}
# ‘sigmoid’, ‘precomputed’ - jeszcze nie potrafię wlasciwie uzywac, wiec pomijam
# "ovo" - "one-versus-one"; "ovr" - "one-versus-rest" (bez zmian)
clf = make_pipeline(MinMaxScaler(), svm.SVC(kernel = "poly"))
clf.fit(X_train,
        y_train)
clf.get_params()

print(confusion_matrix(y_val, clf.predict(X_val)))
print(classification_report(y_val, clf.predict(X_val)))

# =============================================================================
# (rbf,ovo)
#               precision    recall  f1-score   support
# 
#            1       0.92      0.95      0.94        88
#            2       0.93      0.92      0.92        95
#            3       0.88      0.95      0.92       109
#            4       0.98      0.88      0.93       108
# 
#     accuracy                           0.93       400
#    macro avg       0.93      0.93      0.93       400
# weighted avg       0.93      0.93      0.93       400
# =============================================================================
# (linear,ovo)
#               precision    recall  f1-score   support
# 
#            1       0.94      0.94      0.94        88
#            2       0.91      0.93      0.92        95
#            3       0.92      0.93      0.92       109
#            4       0.96      0.94      0.95       108
# 
#     accuracy                           0.93       400
#    macro avg       0.93      0.93      0.93       400
# weighted avg       0.93      0.93      0.93       400
# =============================================================================
# (poly,ovo)
#               precision    recall  f1-score   support
# 
#            1       0.96      0.98      0.97        88
#            2       0.97      0.94      0.95        95
#            3       0.91      0.96      0.93       109
#            4       0.97      0.92      0.94       108
# 
#     accuracy                           0.95       400
#    macro avg       0.95      0.95      0.95       400
# weighted avg       0.95      0.95      0.95       400
# 
# =============================================================================

# Jak dla mnie, jądro wielomianowe jest zwyciezcą (ponizej sciagawka jak czytac macierz)
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
# [[ 86   2   0   0]  <- to wszystko zaklasyfikowano jako 1
#  [  4  89   2   0]  <- to wszystko zaklasyfikowano jako 2
#  [  0   1 105   3]  <- to wszystko zaklasyfikowano jako 3
#  [  0   0   9  99]] <- to wszystko zaklasyfikowano jako 4
#     1   2   3   4   <- kolejno to sa odpowiednie (prawdziwe) kategorie

print("Confusion matrix\n",confusion_matrix(y_test, clf.predict(X_test)),"\n")
print(classification_report(y_test, clf.predict(X_test)))
# Confusion matrix
#  [[99  1  0  0]
#  [ 5 93  3  0]
#  [ 0  5 90  1]
#  [ 0  0  5 98]] 

#               precision    recall  f1-score   support

#            1       0.95      0.99      0.97       100
#            2       0.94      0.92      0.93       101
#            3       0.92      0.94      0.93        96
#            4       0.99      0.95      0.97       103

#     accuracy                           0.95       400
#    macro avg       0.95      0.95      0.95       400
# weighted avg       0.95      0.95      0.95       400
# %% XGBoost



