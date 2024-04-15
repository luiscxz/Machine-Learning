# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:49:58 2023
# algoritmo que hace validacion cruzada
@author: anboo
"""
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# definiendo URL
url = 'https://raw.githubusercontent.com/luiscxz/ML_Py_23/main/data/casas_boston.csv'
file = pd.read_table(url,delimiter=',')
# seleccionando variables independientes
variables_independientes =  [col for col in file.columns if col not in ['MEDV', 'RAD']]
variables_independientes = file[variables_independientes]
# seleccionando vairbales objetivo
variable_objetivo = file.MEDV # MEDV es el precio promedio
# procedemos a definir un modelo de regresion lineal
modelo_regresion = LinearRegression()
#procedemos a ajustar el modelo (construye la regresion) 
modelo_regresion.fit(X=variables_independientes, y= variable_objetivo)
# procedemos a calcular alpha y beta
alpha = modelo_regresion.intercept_
beta = modelo_regresion.coef_
#  procedemos a obtener las predicciones del modelo
file['predicciones']=modelo_regresion.predict(variables_independientes)
# procedemos a elimnar la columna RAD
file.drop('RAD',axis=1,inplace=True)
#%%
'''Función para evaluar el modelo. Sus argumentos son:
    - independientes: tabla de columnas predictoras (es la tabla azul)
    - nombre_columna_objetivo: es el nombre de la columna objetivo de la tabla original
    - tabla_full: es la tabla completa del comentario anterior'''

def evaluar_regresion(variables_independientes,file,beta):
    n = len(file.index) # n es el numero de filas
    k = len(beta)
    mae = metrics.mean_absolute_error(variable_objetivo,file['predicciones'])
    rmse = np.sqrt(metrics.mean_squared_error(variable_objetivo,file['predicciones'])) # desviacion estandar
    r2 = metrics.r2_score(variable_objetivo,file['predicciones'])
    r2_adj = 1-(1-r2)*(n-1)/(n-k-1)
    return {"r2_adj":r2_adj,"mae":mae,"rmse":rmse}
#%%
evaluar_regresion(variables_independientes,file,beta)
#%% entrenamiento y prueba del modelo
from sklearn.model_selection import train_test_split
"""
procedemos a dividir en entrenamiento y prueba.
dataprueba = al 33% del total de datos
random_state=13. Este valor puede ser cualquier número entero no negativo.
test_size : es el procentaje de datos aleatorios 
"""
# procedemos a separar los datos de prueba y de entrenamiento
indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                variable_objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=42)
#%%
# concatenando dataframe  que tiene los datos de entrenamiento
entrenamiento = pd.concat([indepen_entrenamiento,objetivo_entrenamiento],axis=1)
prueba = pd.concat([indepen_prueba,objetivo_prueba], axis=1)
# procedemos a entrenar el modelo
modelo_entrenamiento = LinearRegression()
modelo_entrenamiento.fit(X=indepen_entrenamiento, y=objetivo_entrenamiento)
# obteniendo predicciones con datos de entrenamiento
entrenamiento['predicciones'] = modelo_entrenamiento.predict(indepen_entrenamiento)
# procedemos a predecir el valor de las casas cpn los datos de prueba
prueba['predicciones'] = modelo_entrenamiento.predict(indepen_prueba)
# evaluamos el rendimiento del modelo
n = len(prueba.index) # n es el numero de filas
k = len(beta)
mae = metrics.mean_absolute_error(objetivo_prueba,prueba['predicciones'])
rmse = np.sqrt(metrics.mean_squared_error(objetivo_prueba,prueba['predicciones'])) # desviacion estandar
r2 = metrics.r2_score(objetivo_prueba,prueba['predicciones'])
r2_adj = 1-(1-r2)*(n-1)/(n-k-1)
print(r2_adj)

"""
nota: en una buena regresion no tiene sentido que el algoritmo funcione mejor 
en la prueba que en el entrenamiento, esto debe verse al comprar el rmse. y debe suceder que: 
    el algoritmo se equivoque mas en la prueba que en el entrenamiento
para evitar esto, debemos hacer validacion cruzada
"""
#%% 
from sklearn.model_selection import cross_val_score
modelo_regresion_validacion = LinearRegression()
# estableciendo funcion que va a realizar la validacion cruzada
# scoring= raiz cuadrada del error cuadratico medio rmse (neg_root_mean_squared_error)
#cv = cuantas validaciones quiero realizar
cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                variable_objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=10)
# el promedio de los 10 numeros, debe ser muy parecido al error verdadero rmse
#-------------------------------
# calculando el promedio de los errores y almacenandolos en una lista 
rmse_validacion =[-cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                variable_objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=x).mean() for x in range(10,150)]
#  creando diccionario q
evaluacion_cruzada = {"particiones":list(range(10,150)),
                      "rmse_validacion":rmse_validacion}
evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)
# procedemos a ver la estabilizacion del modelo en base a las particiones
(ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )

