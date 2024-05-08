# -*- coding: utf-8 -*-
"""
Detección de anomalías 
En este código analizamos un datased de transacciones de tarjetas de crédito
el objetivo es encontrar aquellas transacciones anomalas sospechosas de 
fraude o un error

@author: Luis A. Gacía
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
# accediendo a la ruta donde están los datos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo taba de transacciones
file = pd.read_csv('CC_GENERAL.csv');
# procedemos a ver los tipos de columnas que tenemos
file.dtypes
# para este ejercicio no me sirve usar el id del cliente
id_cliente = file['CUST_ID']
# seleecionando todas las columnas a usar
dataset = file.select_dtypes(['float','int'])
# procedemos a buscar las columnas donde faltan datos
dataset.columns[dataset.isnull().any()]
# estos valores nan, los voy a volver 0, ya que puede ser un valor atípico 
# y si los relleno con el promedio, estaria suavisando los datos
dataset = dataset.fillna(0)
#%% Implementación del modelo DBSCAN
"""Hemos aprendido que el algoritmo DBSCAN no asigna un cluster a todos los puntos
si no que aquellos puntos que están más separados del resto se etiquetan
automáticamente como valores extremos
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
""" Al usar un algoritmo de clustering basado en densidad tenemos que 
estandarizar los datos.
Aremos una estandarización estandar resta la media y divide entre la desviación
"""
dataset_estandarizado = StandardScaler().fit_transform(dataset)
# Convirtiendo a dataframe
dataset_estandarizado = pd.DataFrame(dataset_estandarizado,
                                     columns=dataset.columns,
                                     index = dataset.index)