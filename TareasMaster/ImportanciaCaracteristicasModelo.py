# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:02:33 2024

Nota: Este algoritmo debe ejecutarse en el entorno pi_skater con las librerías:
•	Pandas 1.3.5
•	Numpy 1.21.5
•	Sklearn 1.2.2
•	Eli5 0.13.0
•   Matplotlib 3.4.3
•   Seaborn 0.11.2
Algoritmo para ver la importancia de las caracteristicas de un modelo
@author: Luis A. García 
"""
# Importando librerías necesarias
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#accediendo al archivo
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\tareas entregadas\\modulo 5.5 act1')
# leyendo archivo
data = pd.read_csv('FIFA 2018 Statistics.csv')
# creando serie booleana que pone True donde existe la palabra Yes
y = (data['Man of the Match'] == "Yes")
# obteniendo nombre de las columnas de tipo int64
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
# defiendo variables independientes
X = data[feature_names]
# dividiendo en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# creando el modelo de bosques 
my_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
#%% Importancia de la permutación (Permutation Importance)
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
# calculando la importancia de cada caracteristica del modelo
perm = PermutationImportance(my_model, random_state = 1).fit(X_test, y_test)
df_weights = eli5.explain_weights_df(perm, feature_names=X_test.columns.tolist())
# se multiplica la desviación estandar por 2 para que sea igual al html que no se puede visualizar en spyder.
df_weights['2 std']=df_weights['std']*2
print(df_weights)
"""
• Las cualidades hacia lo mejor son las características más esenciales, y aquellas
    hacia la base son las menos importantes.
• El número principal en cada fila muestra cuánto disminuyó el rendimiento del 
    modelo con una mezcla aleatoria (en este caso, utilizando "precisión" como métrica de rendimiento).
• Como ocurre con la mayoría de las cosas en el ámbito de la ciencia de datos, 
    hay cierta aleatoriedad en el cambio de rendimiento definitivo al barajar los datos. 
• Medimos la cantidad de aleatoriedad en nuestra estimación de importancia de permutación 
    repitiendo el proceso con varias combinaciones. El número después del signo ± indica 
    cómo varió el rendimiento de una reorganización a otra.
• Ocasionalmente observaremos valores negativos para las importancias de permutación. 
    En esos casos, las predicciones sobre los datos barajados (o ruidosos) resultaron 
    ser más precisas que los datos reales. Esto ocurre cuando la característica no era 
    relevante (debería haber tenido una importancia cercana a 0), pero la casualidad hizo 
    que las predicciones sobre los datos barajados fueran más precisas. Esto es más 
    común en conjuntos de datos pequeños, como el de este modelo, porque hay más 
    espacio para la suerte o la casualidad.
• En nuestro ejemplo, la característica más importante fue los goles marcados. 
    Eso parece razonable. Los aficionados al fútbol pueden tener cierta intuición 
    sobre si las clasificaciones de diferentes variables son sorprendentes o no.
"""