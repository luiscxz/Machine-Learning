# -*- coding: utf-8 -*-
"""
Algoritmo que usa diferentes modelos para detectar cancer de mama

@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import pandas as pd
import numpy as np
# Accediendo a donde están los datos
os.chdir('D:\\3. Cursos\\9. Data machine learning\\CancerDeMama')
# Leyendo archivo csv
file = pd.read_csv('Cancer_Data.csv')
#eliminando columna vacía
file =file.drop('Unnamed: 32',axis=1)
#%% Explorando datos
# buscando cuantos grupos tiene el dataframe segun la columba objetivo
tiposCancer = file.groupby(by=['diagnosis'],dropna = False).agg(
    Tipo=('diagnosis','count'),
    Porcentaje = ('diagnosis',lambda x: (len(x)/len(file))*100)
    ).reset_index()
""" Dado que:
    Registros de cancer B = 62,74%
    Registros de cancer M = 37.25%
    Se observa que las etiquetas estan desbalanceadas 
"""
# buscando filas con valores faltantes
filasNull = file[file.isnull().any(axis=1)]
#-- Realizando diagrama de cajas y bigotes 
columnas =file.columns.to_list()
boxplotChl = file.boxplot(column= columnas[2:31],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Variables', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('medidas', fontsize=20, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
""" Con el diagrama de cajas se pudo observar los siguiente:
    area_mean: 0 hasta 2500
    area_se: 0 hasta 520
    area_worst : 0 a casi 5000
    Lo que indica que estas columnas generan más peso, entonces debemos
    probar normalizando o estandarizando 
"""
#%% Normalización o estandarización de los datos
from sklearn import preprocessing
# Separando en variables dependientes e independientes
objetivo = file['diagnosis']
independientes = file.loc[:,~file.columns.isin({'id','diagnosis'})]
# conservando nombre de las columnas del df independientes
columnas_ind = independientes.columns
# convirtiendo dataframe a array
independientes = np.array(independientes)
# Creando escalador
normalizador = preprocessing.MinMaxScaler()
# Normalizando  con MinMax: resta el minimo y divide entre (max-min)
independientes = normalizador.fit_transform(independientes)
# convirtiendo a dataframe
independientes = pd.DataFrame(independientes,
                              columns=columnas_ind)
# Graficando boxplot para datrafame normalizado
boxplotChl = independientes.boxplot(column= columnas[2:31],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
boxplotChl.set_title('Datos normalizados', fontsize=20, fontweight='bold')
# Procedemos a codificar las variables Categoricas de forma manual
objetivo = objetivo.replace({'M':1,'B':0})
#%%  Hyper-parámetros
# Importando librerías de modelos a usar
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
"""
Usaré los modelos: 
    KNN
    SVM
    Arboles
    Regresión Logística
    
"""
# Creando función que evalua los modelos mediante validación cruzada
def evaluar_modelo(estimador, independientes, objetivo):
    resultados_estimador = cross_validate(estimador, independientes, objetivo,
                     scoring="f1_micro", n_jobs=-1, cv=5)
    return resultados_estimador
