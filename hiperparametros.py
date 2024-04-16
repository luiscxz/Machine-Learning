# -*- coding: utf-8 -*-
"""
Este algoritmo es para optimización de hiperparámetros
En este caso se esta utlizando para decidir que metodo de clasificación supervisada
me conviene usar y elegir los hiperparámetros.
@author: Luis A. García 
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
import time
from plotnine import *

# estableciendo ruta donde estan los archivos
ruta = 'E:\\4. repositorios github\\ML_Py_23\\data'
os.chdir(ruta)
# vamos a trabjar con la tabla iris
file = pd.read_csv('datos_iris.csv')
# seleccionando columnas con datos tipo float
var_independientes = file.select_dtypes(['float'])
var_objetivo = file.select_dtypes(['object'])
#%%
from sklearn.model_selection import cross_validate
# creando funcion que evalua los modelos
def evaluar_modelo(estimador, independientes, objetivo):
    # realizando validación cruzada
    resultados_estimador = cross_validate(estimador, independientes, objetivo,
                     scoring="f1_micro", n_jobs=-1, cv=5)
    return resultados_estimador
#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
# procedemos a llamar la funcion evaluar modelo y guardamos los resultados
# en un diccionario
resultados = {}

resultados["knn"] = evaluar_modelo(KNeighborsClassifier(),
                                   var_independientes,
                                   var_objetivo['Species'])
resultados["arbol_clasificacion"] = evaluar_modelo(tree.DecisionTreeClassifier(),
                                    var_independientes,
                                    var_objetivo['Species'])
resultados["arbol_aleatorio"] = evaluar_modelo(tree.ExtraTreeClassifier(),
                                    var_independientes,
                                    var_objetivo['Species'])
resultados["msv"] = evaluar_modelo(SVC(),
                                   var_independientes,
                                   var_objetivo['Species'])

#%% Procedemos a crear la funcion para ver los resultados
def ver_resultados(resultados):
    # procedemos a convertir el diccionario a un dataframe
    resultados_df = pd.DataFrame(resultados).T
    # obteniendo nombre de las columnas
    resultados_col = resultados_df.columns
    # procedemos recorrer las columnas mediante ciclo for
    for col in resultados_df:
        """ Dado que cada fila de cada columna contiene la información en 
        el siguiente formato:[0.00399971 0.00400662 0.00399971 0.00399923 0.00400662]
        se procede a cálcular el promedio de estos valores en la fila y todos estos valores
        se reemplazan por un unico valor (el valor promedio)
        """
        resultados_df[col] = resultados_df[col].apply(np.mean)
        """ a cada columna se le consulta el valor máximo, y cada fila de esa
        columna, se divide por el valor maximo. Esto se hace para normalizar los
        datos de las columnas
        """
        resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()       
    return resultados_df
# procedemos a ver los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados y mostrando en pantalla.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
print(resultados_df)
