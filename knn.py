# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:02:13 2024
Algoritmo KNN : K - Nearest Neighbors o K vecinos
Aprendizaje supervisado
# éste código busca predecir a que genero pertenece una pelicula.
@author: Luis A. garcía
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
# estableciendo ruta donde está el archivo.
ruta = 'E:\\4. repositorios github\\ML_Py_23\\data'
os.chdir(ruta)
# leyendo archivo
file = pd.read_csv('datos_peliculas.csv', delimiter=',')
# procedemos a seleccionar el dataframe sin tener en cuenta la columna pelicula y secuela
file2 = file.loc[:,~file.columns.isin(['pelicula','secuela'])]
#%% Implementación del modelo KNN

'''
Vamos a suponer que queremos predecir el género de una película en función
de las otras columnas.
''' 
# Importando librerias necesarias
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# procedemos a mostrar los numeros que representan los generos de las peliculas
unicos_generos = sorted(file2['genero'].unique())
# Procedo a definir la variable objetivo (etiqueta)
objetivo = file2['genero']
# procedo a seleccionar los datos a usar para entrenar el modelo
Variables_independientes = file2.loc[:,~file2.columns.isin(['genero'])]
# procedemos a separar los datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(Variables_independientes,
                                                    objetivo,
                                                    test_size=0.20,
                                                    random_state=2023)
# verificando que en los datos de entrenamiento aparescan todos los generos
# de las peliculas
sorted(Y_train.unique())

#%% Como seleccionar los k vecinos
def clasificadores_knn(k):
    # creando modelo por pesos uniformes y distancias
    knn_uniforme = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_distancias = KNeighborsClassifier(n_neighbors=k, weights="distance")
    # entrenando modelos
    knn_uniforme.fit(X_train,Y_train)
    knn_distancias.fit(X_train,Y_train)
    # Obteniendo predicciones
    preds_uniforme = knn_uniforme.predict(X_test)
    preds_distancias = knn_distancias.predict(X_test)
    # Calculando puntuacion f1
    f1_uniforme = f1_score(Y_test, preds_uniforme, average="micro")
    f1_distancias = f1_score(Y_test, preds_distancias, average="micro")
    return (k,f1_uniforme,f1_distancias)
# procedemos a crear una lista por comprension
clasificacion_evaluaciones =[ clasificadores_knn(k) for k in range(1,151,2)]
# este codigo selecciona vecinos empezando en 1 vecino, luego 3 vecinos y temrina en 149 vecinos
# y ejecuta la funcion 
""" convirtiendo clasificación evaluaciones  en dataframe
"""
clasificacion_evaluaciones = pd.DataFrame(clasificacion_evaluaciones,
                                          columns =['K','f1_uniforme','f1_distancias'])
# Procedemos a convertir la tabla a formato largo
clasificaciones_evaluaciones_tidy = clasificacion_evaluaciones.melt(id_vars=['K'], 
                                                                   var_name='F1_tipo', 
                                                                   value_name='F1_score')
#%%
from plotnine import *
import mplcursors
# procedemos a graficar la tabla en formato largo
(ggplot(data = clasificaciones_evaluaciones_tidy) +
    geom_point(mapping=aes(x="K",y="F1_score",color="F1_tipo")) +
    geom_line(mapping=aes(x="K",y="F1_score",color="F1_tipo"))
)
# habilitar cursor
mplcursors.cursor(hover=True)
#%%
""" Una vez identificado cual modelo obtuvo mejor puntuacion f1
procedemos obtener las predicciones
"""
mejor_clasificador_knn_uniforme = KNeighborsClassifier(n_neighbors=15, weights="uniform")
mejor_clasificador_knn_uniforme.fit(X_train, Y_train)
mejor_preds_uniforme = mejor_clasificador_knn_uniforme.predict(X_test)