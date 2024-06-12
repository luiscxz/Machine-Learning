# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:07:24 2024

@author: Luis A. García
"""
# importando librerias necesarias
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# cargando el conjunto de datos
iris = datasets.load_iris()
# seleccionando variables independientes
feature = pd.DataFrame(iris.data,
                       columns=iris.feature_names)
# Seleccionando solo 1 característica para la regresión (longitud del petalo)
X = np.array(feature['petal length (cm)']).reshape(-1,1)
# Seleccionando variable objetivo
""" encoding de las etiquetas
0 : setosa
1 : versicolor
2 : virginica
"""
target = pd.DataFrame(iris.target,
                      columns=['target'])
y = target['target']
# dividiendo el set de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# creando modelo
model = LinearRegression()
# entrenando modelo
model.fit(X_train, y_train)
# obteniendo las predicciones del modelo
y_pred = model.predict(X_test)
# calculando error 
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")
# Graficando
plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regresión lineal')
plt.xlabel('Longitud del pétalo')
plt.ylabel('Clase')
plt.legend()
plt.show()
#%% Llegan nuevos datos
nuevos_datos = np.array([[1.5], [4.2], [5.4]])
# realizando predicciones
predicciones = model.predict(nuevos_datos)
# Redondear las predicciones al entero más cercano
clases_predichas = np.round(predicciones).astype(int)
#Asegurarse de que las clases predichas están dentro del rango válido (0, 1, 2)
clases_predichas = np.clip(clases_predichas, 0, 2)
print(clases_predichas)