# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:53:48 2022

@author: Bartosz Lewandowski
"""
# %% Packages
import pandas as pd
# %% Import data
def list_conv(x: list):
    for _ in range(len(x)):
        x[_] = float(x[_])
    return x

def txt_to_list(filename: str):
    file = open(filename, "r")
    content = file.read().split("\n")
    file.close()
    return list_conv(content)

Var_av = txt_to_list("Var_av.txt")
Var_LT = txt_to_list("Var_LT.txt")
Var_mass = txt_to_list("Var_mass.txt")
Var1 = txt_to_list("Var1.txt")
Var2 = txt_to_list("Var2.txt")
Var3 = txt_to_list("Var3.txt")
Var4 = txt_to_list("Var4.txt")
Var5 = txt_to_list("Var5.txt")
Var6 = txt_to_list("Var6.txt")
Target = txt_to_list("Target.txt")

data = {'Var_av': Var_av,
      'Var_LT': Var_LT,
      'Var_mass': Var_mass,
      'Var1': Var1,
      'Var2': Var2,
      'Var3': Var3,
      'Var4': Var4,
      'Var5': Var5,
      'Var6': Var6,
      'Target': Target}

df = pd.DataFrame(data)
print(df.head())