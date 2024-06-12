# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:57:26 2024

Aprendizaje supervisado y no supervizado

@author: Luis A. García
"""
# importando librerias necesarias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# accediendo a la ruta donde está el archivo csv
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\actividades\\modulo 5.3 act 1')
# leyendo archivo "column_2C_weka.csv"
data = pd.read_csv('column_2C_weka.csv')
# mostrando la lista de estilos de gráficos disponibles en Matplotlib
print(plt.style.available)
# seleccionando estilo ggplot
plt.style.use('ggplot')
#%% exploración de datos
# obteniendo información del dataframe (columnas, cantidad de datos no nullos y tipo de dato)
data.info()
# obteniendo resulen estadístico del dataframe
resumen = data.describe()
#%% Graficando las clases
"""
creando lista por comprensión que asigna color rojo si la clase es Abnormal
y asigna color verde si la clase es Normal
"""
color_list =['red' if i == 'Abnormal' else 'green' for i in data.loc[:,'class']]
# graficando todas las columnas, exceto la columna 'class'
scatter_matrix = pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolors= "black")
plt.show()
for ax in scatter_matrix.ravel():
    ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right')