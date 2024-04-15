# -*- coding: utf-8 -*-
"""
Algoritmo que implementa:
    Maquinas de soporte Vectorial (MSV) calse 11

@author: Luis A. García
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np 
from plotnine import *
# Estableciendo ruta donde estan los archivos
ruta = 'E:\\4. repositorios github\\ML_Py_23\\data'
os.chdir(ruta)
file = pd.read_csv("datos_iris.csv")
# procedemos a definir las variables independientes, para este caso son todas las
# columnas con datos numericos
independientes = file.select_dtypes(['float'])
# definiendo variable objetivo
objetivo = file.select_dtypes(['object'])
#detectando especies unicas
especies_unicas = objetivo['Species'].unique()
# procedemos a darle valores a variables nominales
objetivo['Species'] = objetivo['Species'].replace({'setosa':0,'versicolor':1,'virginica':2})
# procedo a realizar un conteo de las especies
grupo = objetivo.groupby('Species').agg(
    conteo = ('Species','count')).reset_index()
#-----------------------------------------------------------------------------#
#separando datos de entrenamiento y prueba 
from sklearn.model_selection import train_test_split
"""
X_train: Datos de entrenamiento (independientes)
X_test: Datos de prueba(independientes)
Y_train: Datos de entrenamiento (Objetivo)
Y_test: Datos de prueba (objetivo)
"""
X_train, X_test,Y_train,Y_test = train_test_split(independientes,
                                                  objetivo['Species'],
                                                  test_size=0.3)
#%% Definiendo modelo de maquina de soporte vectorial
from sklearn.svm import SVC 
from sklearn.metrics import f1_score
# procedemos a crear el modelo
estimador_svm = SVC()
# procedemos a entrenar el modelo
estimador_svm.fit(X_train,Y_train)
# procedemos a hacer las predicciones 
predicciones = estimador_svm.predict(X_test)
# procedemos a calcular el f1
f1_score(Y_test , predicciones, average='micro')
#%%
# procedemos a hacer lo mismo pero con validación cruzada 30 veces
from sklearn.model_selection import cross_val_score
cross_val_score(estimador_svm,
                X=independientes,
                y=objetivo['Species'],
                cv=30,
                scoring="f1_micro").mean()
#%% Procedemos a preguntar por los vectores de soporte
estimador_svm.support_vectors_
# procedemos a preguntar, cuantos sirvieron para cada clase
estimador_svm.n_support_
#%% Procedemos a ver como actuan los kernels pero solo usando dos caracteristicas
from mlxtend.plotting import plot_decision_regions # sirve para hacer graficas
# de selección 
# Procedemos a seleccionar las columnas que su nombe inicia con la palabra "Sepal"
X = X_train.filter(like='Sepal')
Y = Y_train
# procedemos a crear los estimadores de MSV 
estimador_svm_lineal = SVC(kernel='linear')
# entrenando modelo
estimador_svm_lineal.fit(X, Y)
#procedo a preguntar el f1 mediante validación cruzada
cross_val_score(estimador_svm_lineal,
                X =X,
                y =Y,
                cv=30,
                scoring="f1_micro").mean()
# procedemos a probar con un kernel polinomico de grado 2
estimador_svm_poli2 = SVC(kernel="poly",degree=2)
estimador_svm_poli2.fit(X,Y)
cross_val_score(estimador_svm_poli2,
                X=X,
                y=Y,
                cv=30,scoring="f1_micro").mean()
# procedemos a probar un kernel polinomial de grado 3
estimador_svm_poli3 = SVC(kernel="poly",degree=3)
estimador_svm_poli3.fit(X,Y)
cross_val_score(estimador_svm_poli3,
                X=X,
                y=Y,
                cv=30,scoring="f1_micro").mean()
# procedo a graficar. tenga en cuenta que este metodo de graficar solo acepta array 
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_lineal)
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_poli2)
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_poli3)
#%% Procedemos a probar los kernel gauseanos
estimador_svm_rbf = SVC(kernel="rbf")
estimador_svm_rbf.fit(X, Y)
cross_val_score(estimador_svm_rbf,
                X=X,
                y=Y,
                cv=30,scoring="f1_micro").mean()
# procedemos a ver como hizo la frontera el kernel gauseano
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_rbf)
# procedemos a ver como funciona con las gamas 
estimador_svm_rbf_0_1 = SVC(kernel="rbf",gamma=0.1)
estimador_svm_rbf_0_1.fit(X, Y)
# procedemos a ver como hizo la frontera el kernel gauseano gama0.1
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_rbf_0_1)

estimador_svm_rbf_10 = SVC(kernel="rbf",gamma=10)
estimador_svm_rbf_10.fit(X, Y)
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_rbf_10)

estimador_svm_rbf_100 = SVC(kernel="rbf",gamma=100)
estimador_svm_rbf_100.fit(X, Y)
plot_decision_regions(X.to_numpy(), Y.values.ravel(), clf=estimador_svm_rbf_100)
"""
Nota: Para gamas muy altas se sobre ajusta el modelo
"""
# vamos a ver los rendimientos para cada una de las gamas
cross_val_score(estimador_svm_rbf,X=X,y=Y,cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_0_1,X=X,y=Y,cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_10,X=X,y=Y,cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_100,X=X,y=Y,cv=5,scoring="f1_micro").mean()