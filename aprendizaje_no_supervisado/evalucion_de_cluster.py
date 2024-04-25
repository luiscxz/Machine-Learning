# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:37:42 2024
Evalución de clusteres
@author: Luis A. García 
"""
# importando librerias necesarias
import os
import pandas as pd
# accediendo a ruta donde están los archivos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo archivo datos iris
file = pd.read_csv('datos_iris.csv')
# seleccionando columnas que su nombre empieza por la palabra "Sepal"
sepal = file.filter(like='Sepal')
#%% Graficando longitud de sepalo con ancho de sepalo
from plotnine import *
(ggplot(data = sepal) +
    geom_point(mapping=aes(x="Sepal_Length",y="Sepal_Width"))
)
