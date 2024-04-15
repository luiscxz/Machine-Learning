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
#%% Importando libreria de validación cruzada
from sklearn.model_selection import cross_validate
# creando funcion que hace validación cruzada para 5 particiones (cv=5) y
# que lo haga para el score=f1
def evaluar_modelo(estimador,X,Y):
    resultados_estimador = cross_validate(estimator, X, Y,
                                          scoring='f1_micro',# se usa por que las clases estan valanceada,
                                          # eso quiere decir que hay 50 flores de una clase y 50 de otra
                                          cv =5,
                                          n_jobs=-1)
# procedemos a crear la función para ver los resultados
def ver_resultados():
    # creando dataframe de pandas llamado resultado y lo trasponemos
    resultados_df = pd.DataFrame(resultados).T
    # obteniendo los nombres de las columnas del dataframe
    resultados_cols = resultados_df.colums
    # procedemos a iterar sobre cada columna del dataframe
    for col in resultados_df:
        # procedemos a calcular el valor maximo de la columna
        maximo = resultados_df[col].max()
        #calculando promedio de la columna y reemplazando todas las filas con el promedio clculado
        resultados_df[col]=resultados_df[col].apply(np.mean)
        # creando nueva columna en el dataframe 
        # procedemos a dividir cada valor de calumna por el valor maximo 
        resultados_df[col+"_idx"] = resultados_df[col]/maximo 
    resultados_df = resultados_df.sort_values(by=["test_score", "fit_time"], ascending=[False, True])
    return resultados_df
#%%
# importando librerias de los modelos de aprendizaje supervisado
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
# procedemos a crear un diccionario donde guardaremos los cinco puntuaciones f1
#correspondientes a las 5 diviciones que definí en la función evaluar modelo 
resultados = {}
# procedemos a evaluar los modelos y guardar los 5 resultados en el diccionaio
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