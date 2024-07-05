# -*- coding: utf-8 -*-
"""
Algoritmo que usa diferentes modelos para detectar cancer de mama

@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Accediendo a donde están los datos
os.chdir('D:\\3. Cursos\\9. Data machine learning\\CancerDeMama')
# Leyendo archivo csv
file = pd.read_csv('Cancer_Data.csv')
#eliminando columna vacía
file =file.drop('Unnamed: 32',axis=1)
#%% Explorando datos
# buscando cuantos grupos tiene el dataframe segun la columba objetivo
tiposCancer = file.groupby(by=['diagnosis'],dropna = False).agg(
    Tipo=('diagnosis','count'),
    Porcentaje = ('diagnosis',lambda x: (len(x)/len(file))*100)
    ).reset_index()
""" Dado que:
    Registros de cancer B = 62,74%
    Registros de cancer M = 37.25%
    Se observa que las etiquetas estan desbalanceadas 
"""
# buscando filas con valores faltantes
filasNull = file[file.isnull().any(axis=1)]
#-- Realizando diagrama de cajas y bigotes 
columnas =file.columns.to_list()
boxplotChl = file.boxplot(column= columnas[2:31],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Variables', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('medidas', fontsize=20, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
""" Con el diagrama de cajas se pudo observar los siguiente:
    area_mean: 0 hasta 2500
    area_se: 0 hasta 520
    area_worst : 0 a casi 5000
    Lo que indica que estas columnas generan más peso, entonces debemos
    probar normalizando o estandarizando 
"""
#%% Normalización o estandarización de los datos
from sklearn import preprocessing
# Separando en variables dependientes e independientes
objetivo = file['diagnosis']
independientes = file.loc[:,~file.columns.isin({'id','diagnosis'})]
# conservando nombre de las columnas del df independientes
columnas_ind = independientes.columns
# convirtiendo dataframe a array
independientes = np.array(independientes)
# Creando escalador
normalizador = preprocessing.MinMaxScaler()
# Normalizando  con MinMax: resta el minimo y divide entre (max-min)
independientes = normalizador.fit_transform(independientes)
# convirtiendo a dataframe
independientes = pd.DataFrame(independientes,
                              columns=columnas_ind)
# Graficando boxplot para datrafame normalizado
boxplotChl = independientes.boxplot(column= columnas[2:31],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
boxplotChl.set_title('Datos normalizados', fontsize=20, fontweight='bold')
# Procedemos a codificar las variables Categoricas de forma manual
objetivo = objetivo.replace({'M':1,'B':0})
#%%  Definimos los hyperparatros
"""
Usaré los modelos: 
    KNN
    SVM
    Arboles
    Regresión Logística
    
"""
# Estableciendo parámetros de busqueda para knn
parametros_knn = {
    "n_neighbors": [1,10,20,30,40,50],
    "p": [1,2,3],
    "weights": ["uniform", "distance"]
}
# estableciendo parámetros de busqueda para SVM
parametros_svm_poly = {
    "degree": [1,2,3,4],
    "kernel": ["poly"]
}

parametros_svm_gauss = {
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["rbf"]
}
 
# Estableciendo parámetros para arbol 
parametros_arbol = {
    "max_depth": list(range(3,6)),
    "criterion": ["gini","entropy"],    
    "class_weight": [None,"balanced"]
    }

parametros_busqueda_svm = {
    "degree": [1,2,3,4],
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["poly", "rbf"]}

""" Nota: El árbol aleatorio no se le hace optimización de hiperparámetros
"""
#%% Escribimos las funciones de evalución de los modelos y visualización de resultados
# Importando librerías de modelos a usar
from sklearn.model_selection import cross_validate
# Creando función que evalua los modelos mediante validación cruzada
def evaluar_modelo(estimador, independientes, objetivo):
    """
    Parameters
    ----------
    estimador : Modelo de aprendizaje supervisado con los mejores parámetros
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
                     scoring="f1_macro", n_jobs=-1, cv=10)
    return resultados_estimador

#%% Creación de modelos a usar
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Definiendo modelos 
estimador_knn = KNeighborsClassifier()
estimador_svm = SVC()
estimador_arbol = tree.DecisionTreeClassifier()
estimador_arbol_aleatorio = tree.ExtraTreeClassifier()
""" Creando modelos para busqueda aleatoria, que encuentra la mejor combinación
de estos parámetros. Internamente RandomizedSearchCV divide los datos en múltiples
pliegues de entrenamiento y validación. Genera combinaciones aleatorias, y para 
cada combinación, entrena el modelo.
"""
knn_Random = RandomizedSearchCV(estimator=estimador_knn, 
                    param_distributions=parametros_knn,
                    scoring="f1_macro", n_jobs=-1,n_iter=4)

svm_poly_Random = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_poly,
                    scoring="f1_macro", n_jobs=-1,n_iter=4)

svm_gauss_Random = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_gauss,
                    scoring="f1_macro", n_jobs=-1,n_iter=4)

arbol_Random = RandomizedSearchCV(estimator=estimador_arbol, 
                    param_distributions=parametros_arbol,
                    scoring="f1_macro", n_jobs=-1,n_iter=4)
#%% Definiendo modelos para busqueda por malla
knn_grid = GridSearchCV(estimator=estimador_knn, 
                    param_grid=parametros_knn,
                    scoring="f1_macro", n_jobs=-1)

svm_grid = GridSearchCV(estimator=estimador_svm, 
                    param_grid=parametros_busqueda_svm,
                    scoring="f1_macro", n_jobs=-1)

arbol_grid = GridSearchCV(estimator=estimador_arbol, 
                    param_grid=parametros_arbol,
                    scoring="f1_macro", n_jobs=-1)
#%% Ajustando los modelos
"""
Se toma el conjunto de datos (independientes y objetivo), se realiza la busqueda
aleatoria de hiperparámetros, y se entrenan y evaluan los modelos con cada combinación
de hyperparámetros y se selecciona la mejor combinación de hiperparámetros basad
 en el F1 score micro
"""
# Ajustando modelos por busqueda aleatoria
knn_Random.fit(independientes, objetivo)
svm_poly_Random.fit(independientes, objetivo)
svm_gauss_Random.fit(independientes, objetivo)
arbol_Random.fit(independientes, objetivo)
# ajustando modelos por busqueda en malla
knn_grid.fit(independientes, objetivo)
svm_grid.fit(independientes, objetivo)
arbol_grid.fit(independientes, objetivo)

#%% Obtención de resultados apartir del mejor estimador y haciendo validación cruzada

resultados = {}
# sección busqueda por malla
resultados["knn_grid"] = evaluar_modelo(knn_grid.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["svm_grid"] = evaluar_modelo(svm_grid.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["arbol_grid"] = evaluar_modelo(arbol_grid.best_estimator_,
                                   independientes,
                                   objetivo)
# sección busqueda aleatoria
resultados["knn_ramdon"] = evaluar_modelo(knn_Random.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["svm_poly_Random"] = evaluar_modelo(svm_poly_Random.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["svm_gauss_Random"] = evaluar_modelo(svm_gauss_Random.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["arbol_Random"] = evaluar_modelo(arbol_Random.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["arbol_aleatorio"] = evaluar_modelo(estimador_arbol_aleatorio,
                                               independientes,
                                               objetivo)
#%%# definiendo función que muestra los resultados 
def ver_resultados(resultados):
    # procedemos a convertir el diccionario a un dataframe
    resultados_df = pd.DataFrame(resultados).T
    # obteniendo nombre de las columnas
    resultados_col = resultados_df.columns
    # procedemos recorrer las columnas mediante ciclo for
    for col in resultados_df:# Col toma los nombres de las columnas
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
#%% observando los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
""" una vez observado e identificado el modelo con mejor puntuación f1_score,
procedemos a identificar sus mejores parámetros
"""
svm_grid.best_params_
#%% Volvemos a crear el modelo y esta vez lo corremos con los mejores parámetros
mejores_params = {'degree': 2, 'gamma': 1.0, 'kernel': 'poly'}
mejor_svm = SVC(**mejores_params,probability=True)
# entrenando el modelo
mejor_svm.fit(independientes, objetivo)
""" Continuar con la llegada de datos nuevos
"""
#%% Procedemos a ver como influyen las características en las decisiones del modelo
import shap
