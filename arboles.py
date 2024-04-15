# -*- coding: utf-8 -*-
"""
Algoritmos de arboles de decision

@author: anboo
"""
'''
Los algoritmos de creación de árboles están en el submódulo de sklearn.tree

En cuanto al tipo de algoritmo para crear árboles, scikit-learn usa una versión
optimizada del algoritmo CART (Classification and Regression Trees), que permite 
usar árboles de decisión tanto para problemas de clasificación como de regresión.
'''
# importando librerias necesarias
import pandas as pd
import os
import numpy as np
# estableciendo ruta
ruta = 'E:\\4. repositorios github\\ML_Py_23\\data'
os.chdir(ruta)
# leyendo archivo titanic
file = pd.read_csv('titanic.csv')
# Procedemos a identificar las columnas con datos string
Columnas_categoricas = file.select_dtypes(['object'])
# procedemos a identificar las columnas con datos int y float
pasajeros = file.select_dtypes(['int','float'])
# convirtiendo columnas categoricas a variables dummy
dummi_categoricas = pd.get_dummies(Columnas_categoricas).astype(int)
# concatenando horizontalmente las columnas int,float y dummi
pasajeros = pd.concat([pasajeros,dummi_categoricas],axis=1)
# debido a que tenemos valores faltantes en la columna edad. los procedemos
# a llenar con el promedio
pasajeros['edad'] = pasajeros['edad'].fillna(pasajeros['edad'].mean())
# procedemos a definir las variables independientes
independientes = pasajeros.loc[:,~pasajeros.columns.isin({'superviviente'})]
# definiendo variable objetivo
objetivo = pasajeros['superviviente']
#%% entrenamiento del modelo
# Importando librerias necesaria para arbol de decisión
from sklearn import tree
from sklearn.model_selection import cross_val_score
# Procedemos a crear el arbol
arbol = tree.DecisionTreeClassifier()
# cargando datos al arbol
arbol.fit(independientes,objetivo)
# procedemos a realizar validación cruzada. cuando se realiza validación cruzada
# no separamos en entrenamiento y prueba.
cross_val_score(arbol,
                independientes,
                objetivo,
                scoring="roc_auc",
                cv=40).mean() # cv: haga 40 veces la separación entre entrenamiento y prueba
# Nota: la validación cruzada internamente separa en entrenamiento y prueba
#%% Procedemos a dibujar el arbol de desición usando graphviz
import graphviz
# accediendo a la ruta donde esta instalado graphviz
os.environ["PATH"] =os.pathsep + "C:\\Program Files\\Graphviz\\bin"
# definiendo función que dibuja arbol de desición
"""
out_file = None : Que no genere ningun archivo
feature_names: pasajeros.drop("superviviente",axis=1).columns  : Se usa para 
seleccionar solo los datos independientes del dataframe. en este caso está
eliminando la columna objetivo

class_names = pasajeros.superviviente.astype(str)  : Es la columna objetivo pero 
convertida a string

"""
def dibujar_arbol(arbol):
    dot_data = tree.export_graphviz(arbol, out_file=None,
                                    feature_names=pasajeros.drop("superviviente",axis=1).columns,
                                    class_names = pasajeros.superviviente.astype(str),
                                    filled=True
                                    )
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("arbol",view=True)
    
dibujar_arbol(arbol)
#%% Variables importantes del arbol de desición
# Procedemos a ver las variables importantes del arbol
Var_importantes = sorted(zip(
    arbol.feature_importances_,
    pasajeros.drop("superviviente", axis=1)), 
    reverse = True)
#%% Supongamos que llega un nuevo pasaje y queremos que el arbol decida si 
# Sobrevive o no.
pasajero_nvo = pd.DataFrame(
    {'clase_billete': [3.],
     'edad': [22.],
     'n_hermanos_esposos': [1.],
     'n_hijos_padres': [0.],
     'precio_billete': [7.25],
     'genero_hombre': [1.],
     'genero_mujer': [0.],
     'puerto_salida_C': [0.],
     'puerto_salida_Q': [0.],
     'puerto_salida_S': [1.]
     }
    )
# usando el arbol de desición vamos a predecir si es pasaje se salva o no
sobrevive = arbol.predict(pasajero_nvo)
#%% Procedemos a hacer un ejemplo controlando la profundidad (cantidad de nodos)
arbol_simple = tree.DecisionTreeClassifier(max_depth=3)
# cargando datos al arbol
arbol_simple.fit(independientes,objetivo)
# haciendo validación cruzada
cross_val_score(arbol_simple,
                independientes,
                objetivo,
                scoring="roc_auc",
                cv=40).mean() # cv: haga 40 veces la separación entre entrenamiento y prueba
# Procedemos a dibu8jar el arbol
dibujar_arbol(arbol_simple)
#%% Procedemos a ver que valor de profundidad maximo me da un mejor cross_val_score
datos = []
for k in range(3,21):
    arbol_simple = tree.DecisionTreeClassifier(max_depth=k)
    arbol_simple.fit(independientes,objetivo)
    calificacion =cross_val_score(arbol_simple,
                    independientes,
                    objetivo,
                    scoring="roc_auc",
                    cv=40).mean()
    # agregando datos al diccionario
    datos.append({'k': k, 'puntaje': calificacion})
puntaje = pd.DataFrame(datos)
#%% Procedemos a hacer un ejemplo pero con balanceo
arbol_balanceado = tree.DecisionTreeClassifier(max_depth=3, class_weight="balanced")
arbol_balanceado.fit(pasajeros.drop("superviviente", axis=1), pasajeros.superviviente)
dibujar_arbol(arbol_balanceado)

cross_val_score(arbol_balanceado, pasajeros.drop("superviviente", axis=1), 
                pasajeros.superviviente, scoring="roc_auc", cv=10).mean()
#%%

'''
Además del algoritmo CART para generar árboles, scikit-learn también proporciona 
una clase de arboles llamada ExtraTreeClassifier, o Extremely Random Trees 
(Árboles Extremadamente Aleatorios). En estos árboles, en lugar de seleccionar 
en cada nodo la párticion que proporciona la mayor ganancia de información, 
¡se decide una partición al azar!. no funciona bien
'''

arbol_aleatorio = tree.ExtraTreeClassifier(max_features=1)
arbol_aleatorio.fit(pasajeros.drop("superviviente", axis=1), pasajeros.superviviente)

dibujar_arbol(arbol_aleatorio)

cross_val_score(arbol_aleatorio, pasajeros.drop("superviviente", axis=1), 
                pasajeros.superviviente, scoring="roc_auc",
                cv=10).mean()