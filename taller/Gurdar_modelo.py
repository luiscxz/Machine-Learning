# -*- coding: utf-8 -*-
"""
Practica de analisis de componentes
En este taller se leen las tablas mnist_pixcel y mnist_clases
Cada fila de la tabla mnist_pixcel corresponde a un número
Cada fila de la tabla mnist_clases es la etiqueta del número, osea que numero es.
@author: Luis A. García
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# accediendo a la ruta donde estan los archivos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo información
pixeles = pd.read_csv('mnist_pixeles.csv',header=None)
labels = pd.read_csv('mnist_clases.csv',header=None)
#%% Procedemos a observar que numero es el que esta en la primera fila de la tabla pixeles
# y lo convertimos a array
primer_digito = pixeles.iloc[0].to_numpy()
# dado que la raiz cuadrada de 784 es 28, convertiremos el arrar "primer_digito"
# una matriz de 28x28 y la graficaremos
primer_digito = primer_digito.reshape(28,28)
# graficando
plt.imshow(primer_digito,cmap="Greys")
# verificando en las etiquetas, que número es el que estoy viendo en la imagen
labels.iloc[0]
#%% Analisis de valanceo de las clases
# Procedemos a ver como estan distribuidos las etiquitas (numeros) en la tabla labels
"""contando cuantos elementos hay en la tabla labels, los multiplicamos *100
y lo dividimos por el total de filas. osea, estoy calculando el porcentaje
de cada clase
"""
labels.value_counts()*100/labels.shape[0]
# según los resultados, la tabla esta desente, valanceada. ya que no exite
# mucha diferencia entre los porcentajes
#%% Implementación de analasis de componentes principales
from sklearn.decomposition import PCA
# implementado modelo para que se quede con las componentes que se juntan el
# 80% de varianza original
modelo_pca = PCA(0.8)
# le pasamos la taba pixeles
pixeles_pca = modelo_pca.fit_transform(pixeles)
#%% implementación del modelo KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint
""" sp_randint es para generar numero aleatorios entre 2 y 10,
al meterlo al diccionario, lo que se le esta pasando es una funcion de distri
bucion de probabilidad
"""
# creando el modelo
clf = KNeighborsClassifier()
#se crea el diccionario para optimización de hiperparametros
busqueda_dist_parametros = {
    "n_neighbors": sp_randint(2,10),
    "p": sp_randint(1,3),
    "weights": ["uniform", "distance"]
}
#%% aplicando optimización de hyperparametros
from sklearn.model_selection import RandomizedSearchCV
# realizando busqueda aleatoria randomizada
busqueda = RandomizedSearchCV(estimator=clf,
                             param_distributions=busqueda_dist_parametros,
                             n_iter=3,# hace 3 iteraciones
                             cv=3,# hace 3 validaciones cruzadas por cada iteración
                             n_jobs=-1,
                             scoring="f1_micro") # se usa f1_micro porque las clases estan valanceadas.
"""#Nota: En la practica es recomendable hacer muchas iteraciones, 30 o más.
# procedemos a pasarle la tabla de componentes principales como variables independientes
 y la tabla labels como variable objetivo
"""
busqueda.fit(X=pixeles_pca, y=labels.values.ravel())
# consultando cual fue la puntucion f1
busqueda.best_score_
# cosultando que parametros dan esa puntuación
mejores_parametros = busqueda.best_params_
#%% corremos el modelo con los mejores parametros encontrados
mejores_params = {'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
# volvemos a correr el modelo con los mejores parámetros
mejor_knn = KNeighborsClassifier(**mejores_params) ## el ** indica que los para
#metros de entrada se la pasan en un diccionario
# entreno el modelo
mejor_knn.fit(pixeles_pca, labels.values.ravel())
# %% NOs llega un dato nuevo
"""Supongamos que nos llega un dato nuevo y necesitamos predecir que numero es
para ello, dado que ya se realizó analisis de componentes principales, lo que debemos
hacer es transformar los nuevos datos, de la siguiente forma:
"""
nuevo_numero = pd.read_csv('mi_numero.csv',header = None)
# transformando los datos
nuevo_pca = modelo_pca.transform(nuevo_numero)
# procedo a predecir los numeros
mejor_knn.predict(nuevo_pca)
# procedemos a verificar que los numero predichos si corresponda
veri_1 = nuevo_numero.iloc[0].to_numpy().reshape(28,28)
veri_2 = nuevo_numero.iloc[1].to_numpy().reshape(28,28)
veri_3 = nuevo_numero.iloc[2].to_numpy().reshape(28,28)
# graficando
plt.imshow(veri_3,cmap="Greys")
#%% Procedemos a guardar el modelo
import pickle
""" 
abriendo archivo modelo_pca.pickle en modo de escritura binara "wb" y 
Serializando el objeto modelo_pca y escribiendo en el archivo abierto.
"""
ruta = 'D:\\3. Cursos\\3. machine learning\\curso machine learning\\codigos realizados en clase\\modelos_guardados'
# Guardando los datos del modelo de componentes prinicipales
with open(ruta+"\\modelo_pca.pickle", "wb") as file:
    pickle.dump(modelo_pca , file)
# procedemos a guardar el mejor modelo knn
with open (ruta+"\\mejor_knn.pickle", "wb") as file:
    pickle.dump(mejor_knn, file)