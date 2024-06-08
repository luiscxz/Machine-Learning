# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:30:06 2024

@author: Luis A. García 
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
# consultando los tipos de columnas que hay el dataframe
features.dtypes
target.dtypes
# convirtiendo columnas categóricas a float
features['CHAS'] = features['CHAS'].astype('float64')
features['RAD'] = features['RAD'].astype('float64')
#%%  Verificando en que número de particiones se equilabra la reducción del error
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
#  creando diccionario que contiene las particiones y los errores promedio
evaluacion_cruzada = {"particiones":list(range(10,150)),
                      "rmse_validacion":rmse_validacion}
# convirtiendo diccionario a dataframe
evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)
# procedemos a ver la estabilización del modelo en base a las particiones
(ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )
#%% Dado que el modelo se estabilizó en 120 particiones procedemos a implementarlo
# Define el modelo de Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
rf = RandomForestRegressor(n_estimators=120, random_state=42)
# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entrenar el modelo con el conjunto de entrenamiento
rf = RandomForestRegressor(n_estimators=120, random_state=42)
rf.fit(X_train, y_train)
# Evaluar el modelo
predictions = rf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")
# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
# Hacer una predicción
new_sample= X_test.iloc[0:1]
prediction = rf.predict(new_sample)
print(f"Predicted price for the sample: ${prediction[0]*1000:.2f}")