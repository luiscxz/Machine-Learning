# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:54:29 2024

@author: anboo
"""

# Importando librerias necesarías
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml 
from plotnine import *
boston = fetch_openml(name='boston', version=1) 
# seleccionando las variables independientes
features = boston.data
# seleccionando variable objetivo
target = boston.target
# consultando los tipos de columnas que hay el daframe
features.dtypes
target.dtypes
# convirtiendo columnas categoricas a floact
features['CHAS'] = features['CHAS'].astype('float64')
features['RAD'] = features['RAD'].astype('float64')
#%%  Verificando en que número de particiones se quilabra la reducción del error
# y la consistencia de los resultados.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# creando estimador
estimador_randomforest = RandomForestRegressor()
# realizando validación cruzada
cross_val_score(estimador_randomforest,
                                    X=features,
                                    y=target, 
                                    scoring="neg_root_mean_squared_error", cv=10)
# calculando el promedio de los errores y almacenandolos en una lista 
rmse_validacion =[-cross_val_score(estimador_randomforest,
                features,
                target,
                scoring = "neg_root_mean_squared_error",
                n_jobs=-1,
                cv=x).mean() for x in range(10,150)]
#  creando diccionario que contiene las paticiones y los errores promedio
evaluacion_cruzada = {"particiones":list(range(10,150)),
                      "rmse_validacion":rmse_validacion}
# convirtiendo diccionario a dataframe
evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)
# procedemos a ver la estabilizacion del modelo en base a las particiones
(ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )
#%% Dado que el modelo se estabilizó en 120 particiones procedemos a implementarlo
# Define el modelo de Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
rf = RandomForestRegressor(n_estimators=120, random_state=42)
cv_results = cross_val_score(rf, features, target, scoring="neg_root_mean_squared_error", cv=120, n_jobs=-1)
rmse_promedio = -np.mean(cv_results)
# Entrenar el modelo con el conjunto de entrenamiento
rf.fit(features, target)

