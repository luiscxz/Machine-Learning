# -*- coding: utf-8 -*-
"""
Código que lee modelos de machine learning guardados
@author: Luis A. García 
"""
# importando librerias necesarias
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# accediendo a la ruta donde están guardados los modelos
os.chdir('D:\\3. Cursos\\3. machine learning\\curso machine learning\\codigos realizados en clase\\modelos_guardados')
# leyendo modelo de componentes prinicipales
with open('modelo_pca.pickle','rb') as file:
    modelo_pca = pickle.load(file)
# leyendo modelo mejor knn
with open('mejor_knn.pickle','rb') as file:
    mejor_knn = pickle.load(file)
#%% Supongamos que nos llega un nuevo sed de datos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
nuevo_numero = pd.read_csv('nuevos_numeros.csv',header = None)
# transformando los datos en componentes principales
nuevo_pca = modelo_pca.transform(nuevo_numero)
# obteniendo prediciones del modelo knn
mejor_knn.predict(nuevo_pca)
# graficando con el fin de validar la informacion predicha por el modelo
plt.imshow(nuevo_numero.iloc[0].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevo_numero.iloc[1].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevo_numero.iloc[2].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevo_numero.iloc[3].to_numpy().reshape(28,28), cmap="Greys")
