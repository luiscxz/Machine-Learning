# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:43:47 2024
Algoritmo para preprosesamiento de los datos
@author: luis A. García
"""
# importando librerias necesarias
import pandas as pd
import os
import numpy as np
# procedemos a establecer la ruta donde estan los datos
ruta = "E:\\4. repositorios github\\ML_Py_23\\data"
# accediendo a la ruta
os.chdir(ruta)
# procedemos a leer el archivo
file = pd.read_csv('datos_procesamiento.csv',delimiter=',')
# procedemos a buscar filas con valores null
filas_null = file[file.isnull().any(axis=1)]
# procedemos a eliminar estas filas
file_sin_null = file.dropna(axis=0)
# procedemos a seleccionar las columnas con valores numericos
fileSinNUll_mum = file_sin_null.select_dtypes(['int','float'])
# seleccionando colmnas string
fileSinNUll_string = file_sin_null.select_dtypes(['object'])
#%% Procedemos a realizar graficos de diagramas de cajas y bigotes
boxplot = fileSinNUll_mum.boxplot(column=['col3'])  
"""
Importando librerias necesarias para estandarizar los datos
"""
from sklearn import preprocessing
# procedemos crear el modelo de escalamiento mediante el metodo estandar
escalador = preprocessing.StandardScaler()
# conviriendo dataframe a array
fileSinNUll_mum_array = np.array(fileSinNUll_mum)
# Procedemos a estandarizar los datos
fileSinNUll_mum_array_Es = escalador.fit_transform(fileSinNUll_mum_array)
# convirtiendo array a dataframe
fileSinNullEst = pd.DataFrame(fileSinNUll_mum_array_Es,
                              columns = fileSinNUll_mum.columns,
                              index =fileSinNUll_mum.index )
# consultado promedio y desviacion estandar
fileSinNullEst.mean()
fileSinNullEst.std()
del escalador,fileSinNUll_mum_array_Es,filas_null
# Procedemos a graficar los datos estandarizados
boxplot = fileSinNullEst.boxplot(column=['col_inexistente1', 'col2', 'col3',
                                          'col_outliers', 'col_outliers2'])  
#%% Procedemos a normalizar los datos mediante escalador minmax
escalador = preprocessing.MinMaxScaler()
fileSinNullNormalizado = escalador.fit_transform(fileSinNUll_mum_array)
# convirtiendo a dataframe
fileSinNullNormalizado = pd.DataFrame(fileSinNullNormalizado,
                                      columns = fileSinNUll_mum.columns,
                                      index = fileSinNUll_mum.index)
# realizando diagrama de cajas para datos normalizados
boxplot = fileSinNullNormalizado.boxplot(column=['col_inexistente1', 'col2', 'col3',
                                          'col_outliers', 'col_outliers2'])
#%% 
""" Sección correspondiente a procesamiento de columnas con string
"""
# primero vamos a seleccionar las variables de texto categoricas (nomilaes y ordinales)
fileSinNUll_stringCategorico = fileSinNUll_string.loc[:,~fileSinNUll_string.columns.isin(['col_texto'])]
# consultado cuantas variables ordinales hay
fileSinNUll_stringCategorico['col_ordinal'].unique()
# procedemos a codificar de forma manual estas variables, ya que las ordinales
# no es recomendable mediante tecnicas de machine learning
fileSinNUll_stringCategorico['col_ordinal_cod'] = fileSinNUll_stringCategorico['col_ordinal'].replace({
    'muy bien':5,'bien':4,'regular':3,'mal':2,'muy mal':1})
# Procedemos a codificar las variables nominales mediante el metodo OneHotEnconde
hot_codificador = preprocessing.OneHotEncoder(sparse=False)
col_nominal = fileSinNUll_stringCategorico['col_categorica']
col_nominal = np.array(col_nominal).reshape(-1,1)
nomila_cod = hot_codificador.fit_transform(col_nominal)
# procedemos a identificar los nombres de las columnas en orden alfabetico
nombre_colum = sorted(fileSinNUll_stringCategorico['col_categorica'].unique())
# creando dataframe
nomila_cod = pd.DataFrame(nomila_cod,
                          columns = nombre_colum,
                          index = fileSinNUll_string.index)
#%% procedemos a procesar los textos
from sklearn import feature_extraction
columna_texto = fileSinNUll_string.col_texto
# procedemos a crear el vectorizador
vectorizador_tf_idf = feature_extraction.text.TfidfVectorizer()
# Procedemos a vectorizar la columna_frases
columna_texto_vect = vectorizador_tf_idf.fit_transform(columna_texto)
# Procedemos a obtener el data frame
frases_vectorizadas=pd.DataFrame(columna_texto_vect .toarray(), columns=vectorizador_tf_idf.get_feature_names_out())
# procedemos a crear el dataframe
datos_procesados = pd.concat([
    fileSinNullEst,
    fileSinNUll_stringCategorico['col_ordinal_cod'],
    nomila_cod,
    frases_vectorizadas
], axis=1)
datos_procesados.to_csv("datos_procesados.csv",index=False)