# -*- coding: utf-8 -*-
"""
Ejemplo de aplicación de metodos ensamblados

@author: Luis A. García
"""
# importando librerias necesarias
import os
import pandas as pd
import numpy as np
# Accediendo a la ruta de los datos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo archivo peliculas csv
file = pd.read_csv('datos_peliculas.csv');
# vamos a verificar como estan valanceados los generos
file['genero'].value_counts()
""" Dado que las clases estan muy desvalanceada, procedemos a generar una columna
objetivo en donde, se ponda cero a las clases desbalanceada y el valor del genero 
a las clases mas valaceadas [1,3,8]
"""
# Añadiendo columna objetivo iniciando en 0
file['objetivo'] = 0 
# asignando valor de genero a objetivo, solo las filas donde genero esta en la lista [1,8,3]
# el modelo solo podra detectar películas del genero 1,3,8
file.loc[file["genero"].isin([1, 3, 8]), "objetivo"] = file["genero"]
file['objetivo'].value_counts()
#%% Seleccionando variables independientes y objetivo
# procedemos a seleccionar todas las columnas excepto "pelicula,año,genero,objetivo"
independientes = file.loc[:,~file.columns.isin(['pelicula','año','genero','objetivo'])]
objetivo = file['objetivo']
# visualizando los tipos de datos que almacenan las columnas de la variables independientes
independientes.dtypes
# procedemos a ver si columna objetivo tiene valanceadas su clases
objetivo.value_counts()*100/objetivo.shape[0]
#%% Dato que todas columnas del dataframe independientes son numéricas, 
# procedemos a realizar un escalado estandar (restar la media y dividir entre la desviación)
from sklearn.preprocessing import StandardScaler
independiente_estand = StandardScaler().fit_transform(independientes)
# convirtiendo a dataframe
independiente_estand = pd.DataFrame(independiente_estand,
                                    columns=independientes.columns,
                                    index = independientes.index)
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
#%% Creación de diccionarios para  Hiper parámetros 
from sklearn.model_selection import RandomizedSearchCV

parametros_knn = {
    "n_neighbors": [10,20,30,40,50],
    "p": [1,2,3],
    "weights": ["uniform", "distance"]
}

parametros_svm_poly = {
    "degree": [1,2,3,4],
    "kernel": ["poly"]
}

parametros_svm_gauss = {
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["rbf"]
}

parametros_arbol = {
    "max_depth": list(range(3,6)),
    "criterion": ["gini","entropy"],    
    "class_weight": [None,"balanced"]
    }
""" Nota: El arbol aleatorio no se le hace optimización de hiperparámetros
"""
#%% Definiendo modelos 
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
estimador_knn = KNeighborsClassifier()
estimador_svm = SVC()
estimador_arbol = tree.DecisionTreeClassifier()
estimador_arbol_aleatorio = tree.ExtraTreeClassifier()
# creando modelos para busqueda aleatoria
knn_grid = RandomizedSearchCV(estimator=estimador_knn, 
                    param_distributions=parametros_knn,
                    scoring="f1_micro", n_jobs=-1)

svm_poly_grid = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_poly,
                    scoring="f1_micro", n_jobs=-1)

svm_gauss_grid = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_gauss,
                    scoring="f1_micro", n_jobs=-1)

arbol_grid = RandomizedSearchCV(estimator=estimador_arbol, 
                    param_distributions=parametros_arbol,
                    scoring="f1_micro", n_jobs=-1)
#%% Procedemos a ajustar los modelos - Hiper Parámetros
knn_grid.fit(independientes, objetivo)
svm_poly_grid.fit(independientes, objetivo)
svm_gauss_grid.fit(independientes, objetivo)
arbol_grid.fit(independientes, objetivo)
#%% almacenando los resultado en un diccionario
resultados = {}

resultados["knn"] = evaluar_modelo(knn_grid.best_estimator_,
                                   independientes,
                                   objetivo)

resultados["svm_poly"] = evaluar_modelo(svm_poly_grid.best_estimator_,
                                   independientes,
                                   objetivo)

resultados["svm_gauss"] = evaluar_modelo(svm_gauss_grid.best_estimator_,
                                   independientes,
                                   objetivo)

resultados["arbol"] = evaluar_modelo(arbol_grid.best_estimator_,
                                   independientes,
                                   objetivo)

resultados["arbol_aleatorio"] = evaluar_modelo(estimador_arbol_aleatorio,
                                               independientes,
                                               objetivo)
# observando los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados y mostrando en pantalla.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
# obteniendo los parametros del modelo que obtuvo mejor puntación
arbol_grid.best_estimator_
#%% Procedemos a ver como se comportan los modelos ensamblados, por defecto
from sklearn.ensemble import BaggingClassifier
# procedemos a definir los modelos con 100 cajas
estimador_bagging_10 = BaggingClassifier(n_estimators=100)
# procedemos a hacer 30 validaciones cruzadas
score_bagging_10 = cross_validate(estimador_bagging_10,
                                  X=independientes,
                                  y=objetivo, 
                                  scoring="f1_micro", cv=30)["test_score"].mean()
score_bagging_10
#%% procedemos a ver como se comportan comportan los modelos ensamblados usando 100 maquina
# vectorial y un kernel rbf
print(BaggingClassifier.__doc__)
estimador_bagging_svm = BaggingClassifier(n_estimators = 100,
                                          estimator = SVC(kernel="rbf"))
# realizando 30 validaciones cruzadas
score_bagging_svm = cross_validate(estimador_bagging_svm,
                                   X=independientes,
                                   y=objetivo, 
                                   scoring="f1_micro", cv=30)["test_score"].mean()
# obteniendo puntaje
score_bagging_svm
#%% procedemos a ver el puntaje que obtiene al usar 100 KNeighborsClassifier
estimador_bagging_knn = BaggingClassifier(n_estimators=100,
                                          estimator=KNeighborsClassifier())
score_bagging_knn = cross_validate(estimador_bagging_knn,
                                   X=independientes,
                                   y=objetivo, 
                                   scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_knn
