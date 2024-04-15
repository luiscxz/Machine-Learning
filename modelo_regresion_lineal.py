# -*- coding: utf-8 -*-
"""
ejemplo de modelo de regresion lineal de complejidad 1

@author: Luis García
"""
# importando librerias necesarias
import pandas as pd
import numpy as np
from siuba import * # de siuba importe todo
from plotnine import *
import sklearn
#-----------------------------------------------------------------------------#
# leyendo archivo como un dataframe, desde github
ruta_archivo_githud ='https://raw.githubusercontent.com/luiscxz/ML_Py_23/main/data/datos_regresion.csv'

file = pd.read_table(ruta_archivo_githud,delimiter=',')
# procedemos a hacer un grafico de correlacion usando plotnine
(ggplot(data=file)+
 geom_point(mapping=aes(x="caracteristica_1", y="valor_real"),color ='red')
 )
# procedemos a definir cuales son las columnas explicativas y cual es la objetivo
objetivo = file.valor_real
variables_independientes = file.caracteristica_1
#%%
from sklearn.linear_model import LinearRegression
# procedemos a definir el modelo
modelo = LinearRegression()
# procedemos a ajustar el modelo (construye la regresion) 
modelo.fit(X =variables_independientes.values[:, np.newaxis], y =objetivo.values[:, np.newaxis])
# calculando alfa y beta
alpha = modelo.intercept_  # esta es el alpha de la regresion
beta = modelo.coef_ # estas son el beta de la regresion

"""
procedemos a obtener las predicciones del modelo.
el modelo toma la beta, se la multiplica a las variables independientes y 
le suma alpha
"""
file['prediccion'] = modelo.predict(variables_independientes.values[:,np.newaxis])

# procedemos a graficar los puntos originales y los predichos
(ggplot(data=file)+
 geom_point(mapping=aes(x="caracteristica_1", y="valor_real"),color ='blue') +
 geom_point(mapping=aes(x="caracteristica_1", y="prediccion"),color ='red')
 )
#%%
# procedemos a evaluar el modelo, osea que tan bueno es el modelo
# calcula el error absoluto medio MAE, es decir, la distancia que hay entre el valor original y el predicho
from sklearn import metrics 
MAE=metrics.mean_absolute_error(file['valor_real'], file['prediccion'])
# calculando error cuadratico medio
MSE = metrics.mean_squared_error(file['valor_real'], file['prediccion'])
# calculando la raiz del error cuadratico medio
np.sqrt(metrics.mean_squared_error(file['valor_real'], file['prediccion']))
# calculando error en cada una de las observaciones
file['errorPrediccion'] = np.absolute(file.valor_real - file.prediccion)
# calculando el coeficiente de determinacion (R2)
coeficiente_determinacion = metrics.r2_score(file['valor_real'], file['prediccion'])
# el que me interesa es el R2 ajustado
k = len(beta)
R2_ajustado = 1 - ((1-coeficiente_determinacion)*(len(file.valor_real)))/(len(file.valor_real - k -1))
# para que el modelo sea bueno R2_ajustado tiene que ser mayor 0.7
