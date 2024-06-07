# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:33:48 2024

@author: Luis A. García
"""
# importando librerias necesarias
import pandas as pd
import numpy as np
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
# graficando diagrama de cajas y bigotes 
columnas =iris.feature_names
boxplotChl = feature.boxplot(column= columnas,
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Variables', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('Medidas', fontsize=20, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
""" Con el diagrama de cajas se pudo observar los siguiente:
    Petal length cm: es la columna con mas varición en los datos 1cm hata 7cm
"""
#%% Definición de hiper parámetros
# Estableciendo parámetros para arbol 
parametros_arbol = {
    "max_depth": list(range(3,6)),
    "criterion": ["gini","entropy"],    
    "class_weight": [None,"balanced"]
    }
#%% Escribimos las funciones de evalución de los modelos y visualización de resultados
# Importando librerías de modelos a usar
from sklearn.model_selection import cross_validate
# Creando función que evalua los modelos mediante validación cruzada
def evaluar_modelo(estimador, independientes, objetivo):
    """
    Parameters
    ----------
    estimador : Modelo de aprendizaje supervisado
    independientes : Dataframe
        Es el dataframe de variables independientes .
    objetivo : Serie
        Contiene las etiquetas del set de datos.

    Returns
    -------
    resultados_estimador : Puntuación F1 Score
        Devuelve las metricas calculadas.
    """
    resultados_estimador = cross_validate(estimador, independientes, objetivo,
                     scoring="f1_micro", n_jobs=-1, cv=5)
    return resultados_estimador
# definiendo función que muestra los resultados 
def ver_resultados(resultados):
    # procedemos a convertir el diccionario a un dataframe
    resultados_df = pd.DataFrame(resultados).T
    # obteniendo nombre de las columnas
    resultados_col = resultados_df.columns
    # procedemos recorrer las columnas mediante ciclo for
    for col in resultados_df:
        """ Dado que cada fila de cada columna contiene la información en 
        el siguiente formato:[0.00399971 0.00400662 0.00399971 0.00399923 0.00400662]
        se procede a cálcular el promedio de estos valores en la fila y todos estos valores
        se reemplazan por un unico valor (el valor promedio)
        """
        resultados_df[col] = resultados_df[col].apply(np.mean)
        """ a cada columna se le consulta el valor máximo, y cada fila de esa
        columna, se divide por el valor maximo. Esto se hace para normalizar los
        datos de las columnas
        """
        resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()       
    return resultados_df
#%% Creación de modelos a usar
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn import tree
# definiendo modelos
estimador_arbol = tree.DecisionTreeClassifier()
estimador_arbol_aleatorio = tree.ExtraTreeClassifier()
""" Creando modelos para busqueda aleatoria, que encuentra la mejor combinación
de estos parámetros. Internamente RandomizedSearchCV divide los datos en múltiples
pliegues de entrenamiento y validación. Genera combinaciones aleatorias, y para 
cada combinación, entrena el modelo.
"""
arbol_Random = RandomizedSearchCV(estimator=estimador_arbol, 
                    param_distributions=parametros_arbol,
                    scoring="roc_auc", n_jobs=-1)
arbol_grid = GridSearchCV(estimator=estimador_arbol, 
                    param_grid=parametros_arbol,
                    scoring="roc_auc", n_jobs=-1)
#%% Ajustando los modelos
"""
Se toma el conjunto de datos (independientes y objetivo), se realiza la busqueda
aleatoria de hiperparámetros, y se entrenan y evaluan los modelos con cada combinación
de hyperparámetros y se selecciona la mejor combinación de hiperparámetros basad
 en el F1 score micro
"""
arbol_Random.fit(feature, target)
arbol_grid.fit(feature, target)
#%% Obtención de resultados apartir del mejor estimador
resultados = {}
resultados["arbol_grid"] = evaluar_modelo(arbol_grid.best_estimator_,
                                   feature,
                                   target)
resultados["arbol_Random"] = evaluar_modelo(arbol_Random.best_estimator_,
                                   feature,
                                   target)
resultados["arbol_aleatorio"] = evaluar_modelo(estimador_arbol_aleatorio,
                                               feature,
                                               target)
# observando los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
#%% usando el mejor modelo encontrado
# obteniendo los parametros del modelo que obtuvo mejor puntación
arbol_grid.best_estimator_
#corremos el modelo con los mejores parametros encontrados
mejores_params = { "max_depth": 3, "criterion": 'gini', "class_weight": None}
mejor_arbol = tree.DecisionTreeClassifier(**mejores_params)
# entrenando el modelo
mejor_arbol.fit(feature, target)
#%% Visualizar el árbol de decisión
import matplotlib.pyplot as plt
from sklearn.tree import  plot_tree
plt.figure(figsize=(20,10))
plot_tree(mejor_arbol, filled=True, feature_names=iris.feature_names,
class_names=iris.target_names)
plt.title('Árbol de decisión para el conjunto de datos Iris')
plt.show()

# nos llega un dato nuevo
muestra = pd.DataFrame(
    {'sepal length (cm)': [5.1],
     'sepal width (cm)': [3.5],
     'petal length (cm)': [1.4],
     'petal width (cm)': [0.2]}
    )
# haciendo predicción
prediccion = mejor_arbol.predict(muestra)
print(f"La especie predicha es: {iris.target_names[prediccion][0]}")

