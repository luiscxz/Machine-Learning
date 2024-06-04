# -*- coding: utf-8 -*-
"""
Preprocesamiento de los datos

@author: Luis A. Garcia
"""
# Importando librerias necesarias
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from plotnine import *
# Establecienso dirección donde se localizan los datos
ruta = "E:\\4. repositorios github\\ML_Py_23\\data"
os.chdir(ruta)
# leyendo archivo de datos 
data = pd.read_table('datos_procesamiento.csv', delimiter=',')
# Consultando cantidad de columnas
data.head()
# Consultando nombre de las columnas
data.columns
# Consultando dimensión del archivo
data.shape
# procedemos a seleccionar las columnas que tienen datos int, float
columas_numericas = data.select_dtypes(['int','float'])
# A estas columnas numéricas le vamos a buscar las filas donde hacen falta datos
filas_vacias = columas_numericas[columas_numericas.isnull().any(axis=1)]
#-----------------------------------------------------------------------------#
""" Procedemos a llenar los espacios vacios.
Ojo. Esto solo llena los espacios vacios con el promedio de las columnas 
del dataframe. Es decir, calcula los promedios de las columnas y si en una 
columna localiza datos faltates entonces los llena con el promedio de esa 
columna
"""
imputador = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean")
columnas_numericas_imputadas = imputador.fit_transform(columas_numericas)
# Convirtiendo el array en dataframe
Col_num_imput = pd.DataFrame(columnas_numericas_imputadas,
                             index=columas_numericas.index,
                             columns = columas_numericas.columns)
"""
Procedemos a estandarizar los datos.
Estandarizar es hacer que el promedio de los datos sea cero y la desviacion sea
1.
estandarizar = (columna[i]-promedio_columna)/desviacion_columna
"""
# calculando el promedio de todas las columnas
mean_columnas = columas_numericas.mean()
# calculando desviación estandar poblacional para todas las columnas.
des_poblacional = columas_numericas.std(ddof=0)
# importando libreria que usaré para estandarizar los datos
from sklearn import preprocessing
# creando el escalador
escalador = preprocessing.StandardScaler()
# estandarizando los datos con el escalador
columas_numericas_estan = escalador.fit_transform(columas_numericas)
# consultando los promedios  y desviaciones estandar de las columnas de los valores estandarizados

columas_numericas_estan.mean(axis=0)
columas_numericas_estan.std(ddof=0,axis=0)
# convirtiendo los datos estandarizados a dataframe
columas_numericas_estan = pd.DataFrame(
    columas_numericas_estan,
    index=columas_numericas.index,
    columns=columas_numericas.columns)
#%% 
# graficando diagrama de caja de la columna col_outliers2
(ggplot(data = Col_num_imput) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
"""
Cuando el gráfico de cajas y bigotes muestra muchos datos atipicos, se recomienda
hacer un escalado robusto y se realiza de la siguiente forma
"""
# definiendo escalador robusto
escalador_robusto = preprocessing.RobustScaler()
# procedemos a estandarizar los datos
columnas_numericas_estan_robusto = escalador_robusto.fit_transform(columas_numericas)
# procedemos a convertir en dataframe
columnas_numericas_estan_robusto = pd.DataFrame(
    columnas_numericas_estan_robusto,
    index=columas_numericas.index,
    columns= columas_numericas.columns)

# graficando los resultados obtenidos

(ggplot(data = columnas_numericas_estan_robusto) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
#%%
# Consultando valores minimos y maximos
columas_numericas.min()
columas_numericas.max()
# creando escalador minmax
escalador_min_max = preprocessing.MinMaxScaler()
# procedemos a normalizar los datos
datos_normalizados = escalador_min_max.fit_transform(columnas_numericas_imputadas)
# procedemos a convertir a dataframe
datos_normalizados = pd.DataFrame(datos_normalizados,
                                  columns = columas_numericas.columns,
                                  index = columas_numericas.index)
# verificando que la normalización se realizara con exito
datos_normalizados.min()
datos_normalizados.max()
# procedemos a graficar los datos normalizados
(ggplot(data = datos_normalizados) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
#%% procesamiento de variables categoricas
# procedemos a seleccionar las variables str
datos_str = data.select_dtypes(['object'])
datos_str = data[['col_categorica','col_ordinal','col_texto']]
# procedemos a seleccionar las columnas categoricas
columnas_categoricas = datos_str.loc[:,~datos_str.columns.isin(['col_texto'])]
# procedemos a buscar cuantos valores unicos hay en la columna col_ordinal
unicos_odinales = columnas_categoricas['col_ordinal'].nunique()
# mostrando esos valores unicos
unicos_odinales = columnas_categoricas['col_ordinal'].unique()
# procedemos a crear la codificación para nuestros valores unicos
"""
muy bien = 5, bien = 4, regular =3, mal = 2, muy mal = 1

"""
# cambiando estos valores en el dataframe = 
columnas_categoricas['codificacion_ordinal'] = columnas_categoricas['col_ordinal'].replace({
    'muy bien':5, 'bien':4, 'regular':3, 'mal':2, 'muy mal':1})
#%% Variables nominales 
# Preparando OneHotEncoder
hot_codificador = preprocessing.OneHotEncoder(sparse_output=False)
"""
OneHotEncoder no recibe una serie, recibe un array 2D, por lo cual, la serie (1D)
debe cambiarse a  2d
"""
col_nominal = columnas_categoricas['col_categorica']
# convirtiendo serie a array 2
col_nominal = np.array(col_nominal).reshape(-1, 1)
# procedemos a codificar 
variables_nomilaes_cod = hot_codificador.fit_transform(col_nominal)
# procedemos a ordenar en orden alfabetico unicas variables nominales
nombre_columnas = sorted(columnas_categoricas['col_categorica'].unique())
# procedemos a crear el dataframe que contiene las variables nominales
df_variables_nomilaes_cod = pd.DataFrame(variables_nomilaes_cod,
                                         columns = nombre_columnas,
                                         index =columnas_categoricas['col_categorica'].index )
#%%
# codificación de texto mediante el metodo CountVectorizer
"""
Usaremos TF-IDF (Frecuencia de Texto - Frecuencia Inversa de Docuemento)

"""
# importando libreria necesaria para codificar texto
from sklearn import feature_extraction
# procedemos a seleccionar la columna del dataframe que contiene frases
columna_frases = data.col_texto
# procedemos a crear el vectorizador
vectorizador_tf_idf = feature_extraction.text.TfidfVectorizer()
# Procedemos a vectorizar la columna_frases
columna_frases_vect = vectorizador_tf_idf.fit_transform(columna_frases)
# Procedemos a obtener el data frame
frases_vectorizadas=pd.DataFrame(columna_frases_vect.toarray(), columns=vectorizador_tf_idf.get_feature_names_out())

