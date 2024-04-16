# -*- coding: utf-8 -*-
"""
Algotimo Hiperparametros aplicado a como se trabaja en la realidad

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# definiendo estimador
estimador_knn = KNeighborsClassifier() 
# visualizando la documentación
print(estimador_knn.__doc__)
# mostrando como se deben usar los parámetros
estimador_knn.get_params()
# creando diccionario que contiene los parámetros
parametros_busqueda_knn = {
    "n_neighbors": [1,10,20,30,40,50],
    "p": [1,2,3],
    "weights": ["uniform", "distance"]
}
# procedemos a contruir los modelos, y a relizar busqueda por malla
knn_grid = GridSearchCV(estimator=estimador_knn, 
                    param_grid=parametros_busqueda_knn,
                    scoring="f1_micro", n_jobs=-1) # ya contiene la validación cruzada
"""n_jobs=-1 hace uso de todos los nucleos de mi máquina
"""
# Procedemos a obtener la hora actual de la coputadora
start_time = time.time()
# procedemos a ajustar los modelos
knn_grid.fit(var_independientes, var_objetivo['Species'])
# capturando hora de finalización
end_time = time.time()
# calculando duración
elapsed_time = end_time - start_time
# imprimiendo mejor puntaje de los modelos
print(knn_grid.best_score_)
# mostrando los parámetros del mejor puntaje
print(knn_grid.best_estimator_.get_params())
#%%
#-----------------------------------------------------------------------------#
# Realizando busqueda aleatoria en knn
#-----------------------------------------------------------------------------#
knn_random = RandomizedSearchCV(estimator=estimador_knn, 
                    param_distributions=parametros_busqueda_knn,
                   scoring="f1_micro", n_jobs=-1, n_iter=10) # n_inter son los 10 
# a elegir aleatoriamente
start_time = time.time()
knn_random.fit(var_independientes, var_objetivo['Species'])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
print(knn_random.best_score_)
print(knn_random.best_estimator_)
#%% Realiando busqueda por malla en maquina vectorial
from sklearn.svm import SVC
estimador_svm = SVC()
print(estimador_svm.__doc__) # consulta la documentación
estimador_svm.get_params() # consulta los paramatros default
# creando nuestro diccionario de parámetros de interes
parametros_busqueda_svm = {
    "degree": [1,2,3,4],
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["poly", "rbf"]
}
##### Aqui voy (continuar aqui...)

#%% funciones para Evalución del modelo mediante validación cruzada
# y obtener los resultados
from sklearn.model_selection import cross_validate
# creando funcion que evalua los modelos
def evaluar_modelo(estimador, independientes, objetivo):
    # realizando validación cruzada
    resultados_estimador = cross_validate(estimador, independientes, objetivo,
                     scoring="f1_micro", n_jobs=-1, cv=5)
    return resultados_estimador

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
#%% almacenando los resultado en un diccionario
resultados = {}
resultados["knn_gridsearch"]= evaluar_modelo(knn_grid.best_estimator_,
                                             var_independientes,
                                             var_objetivo['Species'])
resultados["knn_randomizedsearch"] = evaluar_modelo(knn_random.best_estimator_,
                                                    var_independientes,
                                                    var_objetivo['Species'])
#%% Procedemos a hacer busqueda por malla
# procedemos a ver los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados y mostrando en pantalla.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
print(resultados_df)
