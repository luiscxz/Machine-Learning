# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:37:42 2024
Evalución de clusteres
@author: Luis A. García 
"""
# importando librerias necesarias
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score,silhouette_samples
from plotnine import *
from yellowbrick.cluster import KElbowVisualizer
# accediendo a ruta donde están los archivos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo archivo datos iris
file = pd.read_csv('datos_iris.csv')
# seleccionando columnas que su nombre empieza por la palabra "Sepal"
sepal = file.filter(like='Sepal')
#%% creando función que evalúa el modelo kmedias, normaliza los datos, 
# me entrega la silueta, las etiquetas y 
def constructor_clusters(data,k):
    # procedemos a normalizar los datos
    escalador = preprocessing.normalize(data)
    # convirtiendo datos normalizados a dataframe
    mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=data.index, 
                                      columns=data.columns)
    # creamos el modelo
    k_medias = KMeans(n_clusters = k ,init='k-means++')
    # alimentamos el modelo
    k_medias.fit(mi_data_normalizado_df)
    # le solicitamos las etiquetas al modelo
    Etiquetas = k_medias.labels_
    # pedimos la silueta total
    silueta = silhouette_score(mi_data_normalizado_df,Etiquetas)
    # obteniendo puntaje calinski
    cal_har = calinski_harabasz_score(mi_data_normalizado_df,Etiquetas)
    
    return k, Etiquetas, silueta, cal_har 
#%% Procedeos a evaluar el modelo con la función
"""inicia en el cloúster 2, ya que falla al iniciar en 1
Creando lista que almacena los resultados de la función constructor_clusters
"""
modelos_kmedias = [constructor_clusters(sepal,k) for k in range(2,10)]
# creando dataframe con los resultados del modelo
resultados = pd.DataFrame([(x[0],x[2],x[3]) for x in modelos_kmedias],
                          columns=['K','silueta','calinski_harabasz'])
#%% procedemos a graficar los tiempo, obtener graficamente el mejor k con el graficador de codos
"""modelos = KMeans(n_init=10)
escalador = preprocessing.normalize(sepal)
# convirtiendo datos normalizados a dataframe
mi_data_normalizado_df = pd.DataFrame(escalador, 
                                  index=sepal.index, 
                                  columns=sepal.columns)
visualizer = KElbowVisualizer(modelos, k=(2,10),metric = "silhouette")
visualizer.fit(mi_data_normalizado_df)
visualizer.show()
"""