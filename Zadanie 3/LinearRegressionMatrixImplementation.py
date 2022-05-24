# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:44:37 2022

@author: Bartosz Lewandowski
"""
# %% Packages
import numpy as np
from math import log
# %% Matrix functions
def T(matrix: list):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

"""""
Jeżeli A jest macierzą n x m, a B macierzą typu m x p, to ich iloczyn, AB, jest macierzą o wymiarach n x p.
Jeżeli C = AB, a c_ij oznacza element na pozycji (i,j), to:
c_ij = Sigma_r^m a_ir*b_rj
dla każdej pary i,j dla której 1<=i<=n oraz 1<=j<=p.
"""""
# Wskaż macierz
def AB(A: list, B: list):
# Utwórz macierz C złożoną z samych zer, o wymiarach liczby wierszy z A i liczby kolumn z B.
    if (np.shape(A)[1] != np.shape(B)[0]):
        return "Incorrect matrix dimensions!"
    n, p, m = np.shape(A)[0] ,np.shape(B)[1], np.shape(A)[1]
    C = [[0 for a in range(p)] for a in range(n)]
    # Przechodząc po kolejnych elementach macierzy C, c_ij = Sigma_r^m a_ir*b_rj
    for i in range(n):
        for j in range(len(C[0])):
            for _ in range(m):
                C[i][j] += A[i][_]*B[_][j]
    return C

def Upper(U, b):
    x = []
    ux = 0
    n = len(U)
    x.append(b[n-1]/U[n-1][n-1])
    for k in range(2, n+1):
        for i in range(1, k):
            ux += U[n-k][-i]*x[i-1]
        x.append((b[n-k]-ux)/U[n-k][n-k])
        ux = 0
    return x[::-1]

def Gauss(m, b):
    M = m.copy()
    B = b.copy()
    n = len(M)
    # Utworzenie macierzy uzupełnionej
    for i in range(len(b)):
        M[i] = M[i]+[b[i]]
    # Eliminacja Gaussa
    for i in range(0, n):
        maxEl = abs(M[i][i])
        maxRow = i
        # Poszukiwanie największego elementu w kolumnie i
        for k in range(i+1, n):
            if abs(M[k][i]) > maxEl:
                maxEl = abs(M[k][i])
                maxRow = k
        # Zamiana wierszy z największym elementem
        for k in range(i, n+1):
            temp = M[maxRow][k]
            M[maxRow][k] = M[i][k]
            M[i][k] = temp
        # Zerowanie kolumny
        for k in range(i+1, n):
            c = (-1)*M[k][i]/M[i][i]
            for j in range(i, n+1):
                if i == j:
                    M[k][j] = 0
                else:
                    M[k][j] += c*M[i][j]
    # Rozdzielenie macierzy uzupełnionej
    B = []
    for i in range(n):
        B += M[i][n:]
        M[i] = M[i][:n]
        
    return Upper(M, B)    

def idn(n):
    m=[[0 for x in range(n)] for y in range(n)]
    for i in range(0,n):
        m[i][i] = 1
    return m

def inv_Gauss(m):
    tab = []
    x = idn(len(m))
    for i in range(0, len(m)):
        tab += [Gauss(m, x[i])]
    return tab
# %% Multiple regression
def MultipleRegression(DataFrame, cols: list, target: int):
    Z = list(DataFrame.iloc[:,target]) 
    
    matrix = []
    matrix.append([1]*len(DataFrame))
    
    for _ in cols:
        matrix.append(list(DataFrame.iloc[:,_]))
        
    XTX = AB(matrix,T(matrix)) # Nie ma sensu transponować podwójnie, dlatego jest taki zapis
    
    return XTX, AB(AB(inv_Gauss(XTX),matrix),T([Z]))

def SSE(eps: list):
    sse = 0
    for err in eps:
        sse += err*err
    return sse
   
def SSR(target: list, pred: list):
    mean = sum(target)/len(target)
    ssr = 0
    for x in pred:
        ssr += (x-mean)**2
    return ssr
    
def SST(SSE: float, SSR: float):
    return SSE+SSR

def R_Squared(SSR: float, SST: float):
    return SSR/SST

def R_Squared_Adj(SSE: float, SST: float, variables: int, n: int):
    return 1-SSE/(n-variables)/SST*(n-1)

def calculate_aic(n, mse, num_params):
	aic = n * log(mse) + 2 * num_params
	return aic