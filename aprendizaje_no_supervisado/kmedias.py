# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:06:20 2024
Aprendizaje no supervisado- Kmedias
@author: Luis A. García
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import plotly.express as px
from plotly.offline import plot
# estableciendo ruta donde estan los archivos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
#defiendo archivo a leer
file = pd.read_csv('datos_clientes.csv')
# Procedemos a seleccionar todas las columnas, excepto Id_cliente
file = file.loc[:,~file.columns.isin({'Id_cliente'})]
# procedemos a reemplazar female a 1 y male a 0
file['Genero'] = file['Genero'].replace({'Female':1,'Male':0})
# procedemos a normalizar los datos, esto hace con el fin de quitarle peso a
# algunas columnas que tiene rangos de  0 a 200, mientras que otras van de 0 a 1.
escalador = preprocessing.normalize(file)
# obteniendo dataframe normalizado
file_normalizado = pd.DataFrame(escalador,
                                columns=file.columns,
                                index = file.index)
#%% implementación del modelo de aprendizaje no sumpervizado para ver el mejor k
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
# creando modelo
modelos = KMeans(n_init=10)
#definios el modelo desde 1 hasta 12 closter
visualizer = KElbowVisualizer(modelos, k=(1,12))
# le pasamos la tabla normaliza
visualizer.fit(file_normalizado)
# le pido que haga la gráfica
visualizer.show()
#%% Dado que identificamos el mejor mejor clouster en k = 4
# procedemos a implementar el modelo 
k_medias = KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
# le pasamos los datos al modelo
k_medias.fit(file_normalizado)
# preguntando a que clouster pertenece cada vector, etiquetas
k_medias.labels_
# viendo los centros de cada clouster
k_medias.cluster_centers_
# si me llega un dato nuevo, puedo utilizar la siguiente linea
#k_medias.predict("poner el dataframe de los nuevos clientes")
# procedo a agregar una nueva columna donde agrego los clouster
file['Etiqueta'] = k_medias.labels_.astype(str)
""" Se guarda como str con el fin de que lo tome al momento de graficar como
variable discreta
"""
fig = px.scatter_3d(file, x='Edad', y='Puntuacion_gasto', z='Ingreso_anual',
              color='Etiqueta')
plot(fig)
