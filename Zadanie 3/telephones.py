# -*- coding: utf-8 -*-
"""
Created on Fri May 27 22:17:35 2022

@author: Bartosz Lewandowski
"""
# %% Packages
import pandas as pd

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

desc = telephones_df.describe() # Podejrzany sc_w (min wartosc = 0)

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
# Nalezy dodac +1 do wszyskich wartosci. Powinno to zapobiec przyszlym problemom
# z klasyfikowaniem lub eksploracja danych.
telephones_df['price_range'] = telephones_df['price_range']+1
# =============================================================================

# 3. Multivariate study. We'll try to understand how the dependent variable 
# and independent variables relate.
# =============================================================================
# No need to standardize. Because by definition the correlation coefficient is
# independent of change of origin and scale. As such standardization
# will not alter the value of correlation.
corr = telephones_df.corr()
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
