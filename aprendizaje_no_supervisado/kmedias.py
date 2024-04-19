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