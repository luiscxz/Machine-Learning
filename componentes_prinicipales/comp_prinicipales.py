# -*- coding: utf-8 -*-
"""
Algoritmo de coponentes principales
@author: Luis A. García 
Nota: Antes de hacer análisis de componentes principales, se debe hacer prepro
cesamiento (escalado) si las variables no son dimensionalmente homogeneas, osea tengan 
diferentes dimensiones, ejemplo kg, cm, N, Mpa...
"""
# importando librerias necesarias
import os
import pandas as pd
from plotnine import *
# accediendo a la ruta donde esta los datos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo tabla de cancer de mama
file = pd.read_csv('cancer_mama.csv')
# separamos los datos de la columna objetivo
variables_independientes = file.loc[:,~file.columns.isin(['diagnosis'])]
#%% Sección de preprocesamiento (escalado)
#importando libreria para preprocesamiento.
from sklearn import preprocessing
# creando escalador estadar- osea restamos la media y dividimos 
#entre la varianza de cada una de las columnas
escalador = preprocessing.StandardScaler()
# escalamos el dataframe
escaladas = escalador.fit_transform(variables_independientes)
# convirtiendo a dataframe
var_ind_escaladas = pd.DataFrame(escaladas,
                                 columns =variables_independientes.columns,
                                 index = variables_independientes.index)
# consultando dimensión del dataframe
var_ind_escaladas.shape
#%% Sección de componentes principales
from sklearn.decomposition import PCA
# para este caso vamos a calcular dos componentes principales
modelo_pca = PCA(n_components=2)
# calculando las componentes prinicipales
componentes_principales = modelo_pca.fit_transform(var_ind_escaladas)
# comvirtiendo las componentes prinicipales a dataframe
df_comp_principales = pd.DataFrame(componentes_principales,
                                   columns = ['Comp1','Comp2'])
# verificando cuanta información se almacenó en cada componente.
informacion = modelo_pca.explained_variance_ratio_
# sumando para ver el total de información almacenada
sum(informacion)
# procedemos a agregar la columna objetivo al dataframe que contiene las componentes principales
df_comp_principales = df_comp_principales.assign(diagnosis =file['diagnosis'].astype(str))
# graficando las componentes prinicipales
(    
 ggplot(df_comp_principales) +
    geom_point(mapping = aes(x="Comp1",y="Comp2",color = "diagnosis"),alpha = 0.5)
)


