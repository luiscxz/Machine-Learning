# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:47:49 2024

@author: anboo
"""
# importando liberías necesarias
import os
import pandas as pd
#accediendo a ruta del archivo
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\actividades\\modulo5.5 act 2')
# leyendo archivo
wine_data = pd.read_csv('winequality-red.csv',delimiter=';')
# seleccionando variables independientes
X = wine_data.loc[:,~wine_data.columns.isin(['quality'])]
# seleccionando variable objetivo
y = wine_data['quality']
#%% División de datos en entrenamiento-prueba, optimización de hiperparámetros
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
# diviendo en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# definiendo diccionario de parámetros 
param_grid ={
    'n_estimators':[50,100,200],
    'learning_rate':[0.01,0.1,0.2],
    'max_depth':[3,4,5]
    }
# creando modelo gradiente Boosting
model = GradientBoostingRegressor()
# procedemos a buscar los mejores parámetros
grid_search = GridSearchCV(model, param_grid,cv=5)
grid_search.fit(X,y)
# Mejores parámetros
best_params = grid_search.best_params_
print(f"Mejores parámetros: {best_params}")
#%% Realizando validación cruzada
from sklearn.model_selection import cross_val_score
# usando los mejores parámetros encontrados
best_model = GradientBoostingRegressor(**best_params)
# realizando validación cruzada
cv_score=cross_val_score(best_model, X, y, cv=5,scoring='neg_mean_squared_error')
# calculando el error cuadrático medio.
mse_mean = -cv_score.mean()
print(f"Error cuadrático medio promedio: {mse_mean}")
#%% obteniendo informe de la clasificación del modelo
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
# creando el modelo
modelR = LogisticRegression()
# entrenando el modelo
modelR.fit(X_train,y_train)
# realizando predicciones con el conjunto de prueba
y_pred = modelR.predict(X_test)
# obteniedo informe de clasificación
print(classification_report(y_test, y_pred))