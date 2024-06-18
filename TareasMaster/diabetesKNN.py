# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:10:23 2024

@author: Luis A. García
"""
# importando librerías necesarias
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# accediendo a la ruta del archivo
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\tareas entregadas\\modulo5.4 act1')
# leyendo archivo
diabetes_data = pd.read_csv('diabetes.csv')
# consultado información del dataframe
diabetes_data.info(verbose=True)
# consultando resumen estadistico
resumenEst=diabetes_data.describe()
resumenEstT=diabetes_data.describe().T
"""
La pregunta que surge de este resumen 
¿Puede el valor mínimo de las columnas enumeradas a continuación ser cero (0)? 
En estas columnas, un valor de cero no tiene sentido y por lo tanto indica un valor faltante.
Las siguientes columnas o variables tienen un valor cero inválido:
    Glucose
    BloodPressure
    SkinThickness
    Insulin
    BMI
"""
# realizando copia del data
diabetes_data_copy = diabetes_data.copy(deep = True)
# procedemos a reempalzar los valores zero con nan
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness',
                    'Insulin','BMI']] = diabetes_data_copy[['Glucose',
                                                            'BloodPressure',
                                                            'SkinThickness',
                                                            'Insulin',
                                                            'BMI']].replace(0,np.NaN)
# mostrando la suma de los datos vacios
print(diabetes_data_copy.isnull().sum())
# mostrando histogramas correspondientes a cada columna
p = diabetes_data.hist(figsize = (20,20))
#%% Reemplazando los valores Nan de columnas con los valores promedios de las columna o la mediana
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
# graficando los datos de las columnas sin NaN
p = diabetes_data_copy.hist(figsize = (20,20))
#%% Graficando la cantidad de tipos de datos que tiene el dataframe
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
#%% Graficando diagrama de barras para identificar datos faltantes
import missingno as msno
p=msno.bar(diabetes_data_copy)
# contando la cantidad de diabéticos y nodiabéticos.
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")
#%% Realizando gráfico de pares
from pandas.plotting import scatter_matrix
# Estilo de seaborn
sns.set(style="whitegrid", context='notebook')
p=scatter_matrix(diabetes_data)
for ax in p.ravel():
    ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right')

plt.tight_layout()
#%% Realizando grafico de pares para el la copia del dataframe
p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')
for ax in p.axes.flat:
    ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right')
plt.tight_layout()
#%% Caculando matriz de correlación y realizando mapa de calor a los datos originales
plt.figure(figsize=(12,10)) 
# no se tiene en cuenta los valores nan
p=sns.heatmap(diabetes_data.dropna().corr(), annot=True,cmap ='RdYlGn')
plt.tight_layout()
#%% Caculando matriz de correlación y reliazando mapa de calor al dataframe copy
plt.figure(figsize=(12,10))
# se tiene en cuenta los valore  
p=sns.heatmap(diabetes_data_copy.dropna().corr(), annot=True,cmap ='RdYlGn')
#%% estandarizando datos