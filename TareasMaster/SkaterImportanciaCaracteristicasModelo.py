# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:42:52 2024
Librerías utilizadas en el algoritmo
•	Numpy 1.22.0
•	Pandas 2.2.2
•	Sklearn 1.5.0
•	Xgboost 2.1.0
•	Shap 0.45.1
•	Matplotlib 3.9.0
•   Skater 1.0.2
•   Spyder Notebook
@author: Luis A. García
"""
# importando librerías necesarias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
# estableciendo estilo para graficar
plt.style.use('ggplot')
# Leyendo archivo
data = load_breast_cancer()
# obteniendo descripción del datased
print(data.DESCR)
# obteniendo las definiciones de las etiquetas
pd.DataFrame(data.target_names)
# seleccionando las variables independientes y objetivo
X = pd.DataFrame(data.data,
                 columns =data.feature_names)
y = pd.Series(data.target,
                 name='target')
# procedemos a dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% Entrenamiento del modelo
"""
Creando función se encarga de crear, entrenar y combinar varios modelos de 
aprendizaje automático para construir un clasificador de votación que pueda 
realizar predicciones basadas en la votación de los modelos individuales.
"""
def model_training(X_train, y_train):
    """

    Parameters
    ----------
    X_train : Dataframe
        Variables indepenientes.
    y_train : Serie
        variable objetivo.

    Returns
    -------
    clf1 : Modelo LogisticRegression 
        modelo de regresión logística entrenado.
    clf2 : Modelo RandomForestClassifier 
        modelo de árboles aleatorios entrenado.
    clf3 : modelo GaussianNB 
        modelo de clasificación bayesiana entrenado.
    eclf : Clasificador de votación
        modelo de clasificación combinado.
        
    """
    clf1 = LogisticRegression(random_state=1,max_iter=5000)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    eclf = eclf.fit(X_train, y_train)
    return (clf1,clf2,clf3, eclf)
# aplicando la función para entrenar los modelos
clf1,clf2,clf3, eclf = model_training(X_train,y_train)
#%% Entrenamiento
# creando función entren los modelos
def train_all_model(clf1,clf2,clf3, X_train,y_train):
    clf1 = clf1.fit(X_train, y_train)
    clf2 = clf2.fit(X_train, y_train)
    clf3 = clf3.fit(X_train, y_train)
    models = {'lr':clf1, 'rf':clf2, 'gnb':clf3, 'ensemble':eclf}
    return (clf1,clf2,clf3, models)
# aplicando la función para entrenar los modelos
clf1,clf2,clf3, models = train_all_model(clf1,clf2,clf3,X_train,y_train)
#%% Interpretación de los modelos
"""
creando un conjunto de subgráficos para visualizar la importancia de las 
características (feature importance) de diferentes modelos,lo que permite comparar 
fácilmente cómo diferentes modelos consideran importantes diferentes características en sus decisiones.
"""
f, axes = plt.subplots(2, 2, figsize = (26, 18))

ax_dict = {'lr':axes[0][0],'rf':axes[1][0],'gnb':axes[0][1],'ensemble':axes[1][1]}
interpreter = Interpretation(X_test, feature_names=data.feature_names)

for model_key in models:
    pyint_model = InMemoryModel(models[model_key].predict_proba, examples=X_test)
    ax = ax_dict[model_key]
    interpreter.feature_importance.plot_feature_importance(pyint_model, ascending=True, ax=ax)
    ax.set_title(model_key)
#%% Calculando puntuación F1 score para cada modelo
from sklearn.metrics import f1_score
for model_key in models:
        print("Model Type: {0} -> F1 Score: {1}".
              format(model_key, f1_score(y_test, models[model_key].predict(X_test))))
#%%
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Definir parámetros
grid_resolution = 20  # Define la resolución de la cuadrícula
feature_selection = ['worst area', 'mean perimeter']  # Define las características a analizar

# Definir función de interpretación
def understanding_interaction():
    pyint_model = InMemoryModel(eclf.predict_proba, examples=X_test, target_names=data.target_names)
    interpreter.partial_dependence.plot_partial_dependence(feature_selection,
                                                          pyint_model,
                                                          grid_resolution=grid_resolution,
                                                          with_variance=True)
    plt.show()  # Mostrar el gráfico usando Matplotlib

# Llamar a la función para mostrar los resultados
understanding_interaction()