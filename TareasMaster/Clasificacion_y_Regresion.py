# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:33:48 2024

@author: Luis A. García
"""
# importando librerias necesarias
import pandas as pd
from sklearn.datasets import load_iris
# procedemos a cargar los datos
iris = load_iris()
# seleccionando variables independientes
feature = pd.DataFrame(iris.data,
                       columns=iris.feature_names)
# Seleccionando variable objetivo
""" encoding de las etiquetas
0 : setosa
1 : versicolor
2 : virginica
"""
target = pd.DataFrame(iris.target,
                      columns=['target'])

#%% Exploración de datos
# Buscando filas con valores faltantes en los datos
FilasNull = feature[feature.isnull().any(axis=1)]
# verificando el balance de nlas clases
grupo = target.groupby(by=['target'],dropna = False).agg(
    CantidadRegistros =('target','count'),
    porcentaje = ('target', lambda x: (len(x)/len(target))*100)
    ).reset_index()
"""
Dado que :
    setosa tiene 50 registros 
    versicolor tiene 50 registros
    virginica tiene 50 registros
"""
target = target['target']