# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:32:58 2024
Librerías utilizadas en el algoritmo
•	Numpy 1.26.4
•	Pandas 2.2.2
•	Sklearn 1.5.0
•	Xgboost 2.1.0
•	Shap 0.46.0
•	Matplotlib 3.9.0
Nota: Este algoritmo debe ejecutarse en el entorno shap
@author: Luis A. García
"""
# importando librerías necesarias
from IPython.display import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
import matplotlib.pyplot as plt
from IPython.display import Image
import os
#accediendo al archivo
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\tareas entregadas\\modulo 5.5 act1')
# leyendo archivo 
data = pd.read_csv("FIFA 2018 Statistics.csv")
# creando serie booleana que pone True donde existe la palabra Yes
y = (data['Man of the Match'] == "Yes")
# obteniendo nombre de las columnas de tipo int64
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
# defiendo variables independientes
X = data[feature_names]
# dividiendo en entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=9487)
#%% definiedo diccionario con hiperparámetros para el modelo xgboost
params = {'base_score': 0.5,
          'booster': 'gbtree',
          'colsample_bylevel': 1,
          'colsample_bytree': 1,
          'gamma': 0,
          'learning_rate': 0.05,
          'max_delta_step': 0,
          'max_depth': 3,
          'min_child_weight': 1,
          'missing': None,
          'n_jobs': 1,
          'objective': 'binary:logistic',
          'random_state': 0,
          'reg_alpha': 0,
          'reg_lambda': 1,
          'scale_pos_weight': 1,
          'seed': 0,
          'subsample': 1,
          'verbosity': 0}
# agregando parámetros de evaluación área bajo la curva (auc)
params['eval_metric'] = 'auc'
#%% procedemos a entrenar el modelo
#preparando los datos de entrenamiento y validación para ser utilizados por el modelo XGBoost
d_train = xgboost.DMatrix(X_train, y_train)
d_val = xgboost.DMatrix(X_val, y_val)
# creando lista para monitorear el desempeño del modelo en ambos conjuntos de
# datos durante el procesamiento de entrenamiento
watchlist = [(d_train, "train"), (d_val, "valid")]
# entrenando el modelo
model = xgboost.train(params, d_train, num_boost_round=2000, evals=watchlist, 
                      early_stopping_rounds=100, verbose_eval=10)
# seleccionando la fila 83 de X_train y convirtiendola en DMatrix. Es decir, la preparamos para hacer predicciones
data_for_prediction = xgboost.DMatrix(X_train.iloc[[83],:])
# haciendo predicción utilizando el modelo entrenado
model.predict(data_for_prediction)
#%% Interpretación del modelo mediante shap
import shap 
# creando objeto para interpretar las predicciones del modelo
explainer = shap.TreeExplainer(model)
# calculando los valores shap para todas las observaciones en el conjunto de datos X_train
shap_values = explainer.shap_values(X_train)
#inicializando las dependencias de JavaScript necesarias para las visualizaciones interactivas de SHAP
shap.initjs()
shap.summary_plot(shap_values, X_train)
shap.summary_plot(shap_values, X_train, plot_type='bar')  # Puedes usar 'dot' en lugar de 'bar' si prefieres puntos
plt.show()
#%% Forceplot
"""
 realizando una predicción usando XGBoost y visualizando la importancia de las 
 características en la predicción utilizando SHAP.
"""
data_for_prediction = xgboost.DMatrix(X_train.iloc[[10],:])  # use 1 row of data here. Could use multiple rows if desired
print(f"The 85th data is predicted to be True's probability: {model.predict(data_for_prediction)}")
shap.force_plot(explainer.expected_value, shap_values[10,:], X_train.iloc[10,:],matplotlib=True)
plt.show()
#%%
data_for_prediction = xgboost.DMatrix(X_train.iloc[[83],:])  # use 1 row of data here. Could use multiple rows if desired
print(f"The 83rd data is predicted to be True's probability: {model.predict(data_for_prediction)}")
shap.force_plot(explainer.expected_value, shap_values[83,:], X_train.iloc[83,:],matplotlib=True)
#%%
"""
mostrando la relación entre el logaritmo de las probabilidades de ganar y la probabilidad real de ganar
"""
plt.figure(figsize=(20,8))
xs = np.linspace(-5,5,100)
plt.xlabel("Log odds of winning")
plt.ylabel("Probability of winning")
plt.title("Log odds & prob of winning convert")
plt.plot(xs, 1/(1+np.exp(-xs)))

new_ticks = np.linspace(-5, 5, 11)
plt.xticks(new_ticks)
plt.show()
#%%
"""
genera un gráfico de fuerza utilizando SHAP
este código genera un gráfico que visualiza la contribución de cada característica
en la predicción del modelo para los datos de entrenamiento X_train, utilizando 
los valores SHAP calculados. Cada punto en el gráfico representa una instancia 
de datos, y la posición horizontal indica el impacto de las características en 
la predicción del modelo.
"""
shap.force_plot(explainer.expected_value, shap_values, X_train)
#%%
"""
generando gráfico que visualiza cómo la variable 'Ball Possession %' impacta 
en las predicciones del modelo, según los valores SHAP calculados. El gráfico 
puede mostrar también cómo esta dependencia puede variar en función de la 
interacción con la variable 'Goal Scored'.
"""
shap.dependence_plot('Ball Possession %', shap_values, X_train, interaction_index="Goal Scored")