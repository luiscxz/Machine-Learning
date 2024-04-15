# -*- coding: utf-8 -*-
"""
Algoritmo de regresion logistica

@author: Ing.Oceanográfico Luis A. García
El archivo cancer_mama es un estudio de mujeres que presentaron lunares en la
piel. Basados en caracteristicas de estos lunares se les detecto si tenian 
canser de mama o no.
diagnosis: 1 cancer benigno
           0 cancer maligno
 ######   #######    ####   ######   #######   #####    ####     #####   ##   ##           ####      #####     ####    ####     #####   ######    ####      ####     ##
  ##  ##   ##   #   ##  ##   ##  ##   ##   #  ##   ##    ##     ##   ##  ###  ##            ##      ##   ##   ##  ##    ##     ##   ##  # ## #     ##      ##  ##   ####
  ##  ##   ## #    ##        ##  ##   ## #    #          ##     ##   ##  #### ##            ##      ##   ##  ##         ##     #          ##       ##     ##       ##  ##
  #####    ####    ##        #####    ####     #####     ##     ##   ##  ## ####            ##      ##   ##  ##         ##      #####     ##       ##     ##       ##  ##
  ## ##    ## #    ##  ###   ## ##    ## #         ##    ##     ##   ##  ##  ###            ##   #  ##   ##  ##  ###    ##          ##    ##       ##     ##       ######
  ##  ##   ##   #   ##  ##   ##  ##   ##   #  ##   ##    ##     ##   ##  ##   ##            ##  ##  ##   ##   ##  ##    ##     ##   ##    ##       ##      ##  ##  ##  ##
 #### ##  #######    #####  #### ##  #######   #####    ####     #####   ##   ##           #######   #####     #####   ####     #####    ####     ####      ####   ##  ##
"""
# Importando librerias necesarias
import pandas as pd
import numpy as np
import os
import glob
from plotnine import *
# leyendo archivo cancer de mama
ruta = "E:\\4. repositorios github\\ML_Py_23\\data"
os.chdir(ruta)
archivos = glob.glob('*.csv') # detecta los archivos terminados en csv
# leyendo archivo cancer de mama
file = pd.read_table(archivos[0], delimiter=',')
# cambiando el diagnosis, es decir el cancer malino va a ser 1 y el cancer
# benigno va  a ser 0
file.diagnosis = file.diagnosis.replace({0:1,1:0})
# como mi objetivo es predecir la etiqueta, entonces mi variable objetivo es:
varible_objetivo = file.diagnosis
# definiendo variables independiente
variables_independientes = file.loc[:,~file.columns.isin(['diagnosis'])]
#%% machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# procedemos a separar en entrenamiento y prueba
independiente_entrenamiento, independiente_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                              varible_objetivo,
                                                                                                              test_size =0.3,
                                                                                                              random_state = 42)
# procedemos a definir el modelo, ene este caso hay mucho mas filas que columnas
# entonces utilzo un solver, en este caso newton-cholesky
modelo_regres_logistica = LogisticRegression(solver='newton-cholesky')
# Proceso a entrenar el modelo
modelo_regres_logistica.fit(independiente_entrenamiento,objetivo_entrenamiento)
# Procedemos a obtener las predicciones del modelo, para esto le agregamos 
# los datos de prueba (los independientes) ya que quiero conocer es la prediccion
predicciones =modelo_regres_logistica.predict(independiente_prueba)
# Calculando las probabilidades 
predicciones_probabilidades = modelo_regres_logistica.predict_proba(independiente_prueba)
# procedemos a llamar los objetivos realies
objetivos_reales = objetivo_prueba
#%% 
def tupla_clase_prediccion(objetivos_reales, predicciones):
    return list(zip(objetivos_reales, predicciones))
# Definiendo función que entrega la cantidad de verdaderos positivos
def VP(objetivos_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(objetivos_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])
# Definiendo función que entrega la cantidad de verdaderos negativos
def VN(objetivos_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(objetivos_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
# Definiendo función que entrega la cantidad de falsos positivos     
def FP(objetivos_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(objetivos_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])
# definiendo funcion que entrega los falsos negativos 
def FN(objetivos_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(objetivos_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==0])


print("""
Verdaderos Positivos: {}
Verdaderos Negativos: {}
Falsos Positivos: {}
Falsos Negativos: {}
""".format(
    VP(objetivos_reales, predicciones),
    VN(objetivos_reales, predicciones),
    FP(objetivos_reales, predicciones),
    FN(objetivos_reales, predicciones)    
))
# Creando dataframe que contiene los valores reales y los calculados 
Comparacion = pd.DataFrame({
    'Datos_reales_prueba':objetivos_reales,
    'Predicciones':predicciones})
#%% Sección para las metricas de evaluación
from sklearn import metrics
# definiendo función que calcula la presición
'''Precisión'''
def precision(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fp = FP(clases_reales, predicciones)
    return vp / (vp+fp)
# creando función que modifica las etiquetas si la probabilidad es >= a 0.5
def proba_a_etiqueta(predicciones_probabilidades,umbral=0.5):
    predicciones = np.zeros([len(predicciones_probabilidades), ])
    predicciones[predicciones_probabilidades[:,1]>=umbral] = 1
    return predicciones
# creando función que evalua todos los umbrales, y evalua las metricas para
# cada umbral
def evaluar_umbral(umbral):
    predicciones_en_umbral = proba_a_etiqueta(predicciones_probabilidades, umbral)
    precision_umbral = precision(objetivos_reales, predicciones_en_umbral)
    sensibilidad_umbral = metrics.recall_score(objetivos_reales, predicciones_en_umbral)
    F1_umbral = metrics.f1_score(objetivos_reales, predicciones_en_umbral)
    return (umbral,precision_umbral, sensibilidad_umbral, F1_umbral)
# definiendo cuantos umbrales van a existir
umbrales = np.linspace(0., 1., 1000)
# creando dataframe que contiene las metricas y los umbrales evaluados
evaluaciones = pd.DataFrame([evaluar_umbral(x) for x in umbrales],
                            columns = ["umbral","precision","sensibilidad","F1"])
# procedemos a graficar los resultados obtenidos
(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="sensibilidad",y="precision",color="umbral"),size=0.4)
)

(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="umbral",y="F1"),size=0.1)
)
# Procedo a buscar el valor mámixo de F1
F1_max = evaluaciones.F1.max()
# Procedemos a filtrar el dataframe para obtener las filas donde F1 es igual al
# valor máximo
filtro = evaluaciones.loc[evaluaciones['F1']==F1_max]
