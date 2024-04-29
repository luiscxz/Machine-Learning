# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:31:10 2024
Clusterización por DBSCAN
La complejidad de este modelo está en elegir la pareja eps y  min_samples
que dan la mejor silueta proxima a 1.
@author: Luis A. García
"""
# importando librerías necesarias 
import os
import pandas as pd
import numpy as np
from plotnine import *
# accediendo a ruta de archivos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
file = pd.read_csv('datos_separados.csv')
file2 = pd.read_csv('datos_circulares.csv')
# Graficando datos
(ggplot(data = file) +
     geom_point(mapping=aes(x="columna1",y="columna2"),alpha=0.2) 
)
#%% Sección de Aplicación de DBSCAN para datos separados
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
"""creando modelo con los siguientes paramétros
epsilon (radio) = 3
min samples (minpts): 15 minimo numero de puntos al rededor de un punto  
"""
# creando modelo
dbscan = DBSCAN(eps=3, min_samples=15)
# entrenando modelo y solicitando las etiquetas
etiquetas_dbscan = dbscan.fit(file).labels_
# calculando silueta
silhouette_score(file,etiquetas_dbscan) 
# procedemos a ver cuantos etiquetas unicas existen
np.unique(etiquetas_dbscan)
# agregando etiquetas al dataframe original
file = file.assign(
    label =etiquetas_dbscan)
# procedemos a graficar
(
     ggplot(file) +
          geom_point(mapping=aes(x="columna1",y="columna2",color = "label.astype(str)"),alpha=0.2) 
 
 )
## Tarea: encontrar la pareja eps , min_samples que consigan una silueta mayor a 0.6