# -*- coding: utf-8 -*-
"""
Detección de anomalías 
En este código analizamos un datased de transacciones de tarjetas de crédito
el objetivo es encontrar aquellas transacciones anomalas sospechosas de 
fraude o un error

@author: Luis A. Gacía
"""
# Importando librerias necesarias
import os
import pandas as pd
import numpy as np
# accediendo a la ruta donde están los datos
os.chdir('E:\\4. repositorios github\\ML_Py_23\\data')
# leyendo taba de transacciones
file = pd.read_csv('CC_GENERAL.csv');
# procedemos a ver los tipos de columnas que tenemos
file.dtypes
# para este ejercicio no me sirve usar el id del cliente
id_cliente = file['CUST_ID']
# seleecionando todas las columnas a usar
dataset = file.select_dtypes(['float','int'])
# procedemos a buscar las columnas donde faltan datos
dataset.columns[dataset.isnull().any()]
# estos valores nan, los voy a volver 0, ya que puede ser un valor atípico 
# y si los relleno con el promedio, estaria suavisando los datos
dataset = dataset.fillna(0)
#%% Implementación del modelo DBSCAN
"""Hemos aprendido que el algoritmo DBSCAN no asigna un cluster a todos los puntos
si no que aquellos puntos que están más separados del resto se etiquetan
automáticamente como valores extremos
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
""" Al usar un algoritmo de clustering basado en densidad tenemos que 
estandarizar los datos.
Aremos una estandarización estandar resta la media y divide entre la desviación
"""
dataset_estandarizado = StandardScaler().fit_transform(dataset)
# Convirtiendo a dataframe
dataset_estandarizado = pd.DataFrame(dataset_estandarizado,
                                     columns=dataset.columns,
                                     index = dataset.index)
#%% Procedemos a hacer una busqueda aleatoria para obtimizar los resultados
# de BDSCAN, para esto se hace uso de una distribución de probabilidad continua

from scipy.stats import randint as sp_randint # distribución de probabilidad uniforme discreta
from scipy.stats import uniform  # distribución de probabilidad uniforme continua
# creando diccionario que contiene los parametros a usar
distribucion_parametros = {
    "eps": uniform(0,5),# genera números aleatorios entre 0 y 5, incluyendo decimales ejmplo (0.2,0.1)
    "min_samples": sp_randint(2, 20), # genera números enteros desde el 2 hasta el 19
    "p": sp_randint(1, 3), # genera número correspondiente a la distancia 
}
'''
Un problema que hay con el algoritmo HDBSCAN (o DBSCAN) es que no tiene el 
método predict, por lo tanto no podemos usar el método de busqueda aleatoria 
de scikit-learn.

Sin embargo, podemos desarrollar nuestro propio método de búsqueda con 
ParameterSampler, que es lo que usa scikit-learn para tomar muestras del 
diccionario de búsqueda de hiperparámetros.

Este paso tarda tiempo en ejecutarse
'''
from sklearn.model_selection import ParameterSampler
n_muestras = 30 # probamos 30 combinaciones de hiperparámetros, de las infinitas combinaciones que tengo
n_iteraciones = 20 #para validar, vamos a entrenar para cada selección de hiperparámetros en 3 muestras distintas, 
# y en cada muestra se calcula la silueta, osea que obtengo 3 siluetas para cada combinación de parámetros
pct_muestra = 0.7 # usamos el 70% de los datos para entrenar el modelo en cada iteracion
resultados_busqueda = [] # lista vacía que se usa para almacenar los resultados
lista_parametros = list(ParameterSampler(distribucion_parametros, n_iter=n_muestras))
#%% Calculo de siluetas 
from sklearn.metrics import silhouette_score
# iniciamos ciclo for
for listas in lista_parametros:
    # iniciamos segundo ciclo for que iterea la cantidad de veces que se estableció en n_iteraciones
    for iteration in range(n_iteraciones):
        # iniciamos la lista vacia en cada iteracion
        param_resultados = []
        # obtenemos la muestra aleatoria del dataset normalizado
        muestra = dataset_estandarizado.sample(frac=pct_muestra)
        # obtenemos los clusteres del modelo. se le ingresa la lista de parámetros
        # y se ajusta el modelo con la muestra
        labels_clusters = DBSCAN(n_jobs=-1,**listas).fit_predict(muestra)
        #--------------------- Manejor de errores-----------------------------#
        """ dado que muchas veces DBSCAN no puede asignar clústeres a ciertas
        combinaciones de epsilon y radio, entonces debemos manejar estos errores
        de la siguiente forma:
        """
        try:
            # cuándo se pueda calcular la silueta, agreguela a la lista param_resultados
            param_resultados.append(silhouette_score(muestra,labels_clusters))
            # en caso de que se encuentre con un error durante el camino, lo ignore
        except ValueError:# ignora errores encontrados. veces silhouette_score falla en los casos en los que solo hay 1 cluster
            pass 
    # Calculando el promedio de las siluetas que se pudieron calcular
    puntuacion_media = np.mean(param_resultados)
    # agregando la puntuación media a la lista resultados_busqueda
    resultados_busqueda.append([puntuacion_media,listas])      
"""ordenando lista en función del primer elemento de cada elemento de la lista
de forma descendente y luego selecciono los 5 elementos con los valores más
altos en la lista ordenada
"""
sorted(resultados_busqueda, key=lambda x: x[0], reverse=True)[:5]
#%% Le asignamos los valores que dieron la mejor silueta, mediante un diccionario
mejores_params = {'eps': 4.857511884092619, 'min_samples': 10, 'p': 2}
# corremos el modelo con estos parámetros
clusteres = DBSCAN(n_jobs=-1,**mejores_params)
# obteniendo los clusteres (etiquetas)
cluster_labels = clusteres.fit_predict(dataset_estandarizado)
# contando las etiquetas
pd.Series(cluster_labels).value_counts()
#%% Creando funciones que resumen los resultados
def resumen_cluster(cluster_id):
    cluster = dataset[cluster_labels==cluster_id]
    resumen_cluster = cluster.mean().to_dict()
    resumen_cluster["cluster_id"] = cluster_id
    return resumen_cluster
"""
La función comparar_clusters(*cluster_ids) funciona de la siguiente manera
# recibe los clusteres encontrados, para este caso fue (0,1), el * indica que 
estos valores se empaquetaran como una tupla
"""
def comparar_clusters(*clusteres):
    # creando lista que guarda el resumen
    resumenes = []
    for cluster_id in clusteres:
        resumenes.append(resumen_cluster(cluster_id))
    return pd.DataFrame(resumenes).set_index("cluster_id").T

comparar_clusters(0,-1)
