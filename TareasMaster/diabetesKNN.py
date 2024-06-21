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
# realizando copia de la data
diabetes_data_copy = diabetes_data.copy(deep = True)
# procedemos a reemplazar los valores zero con nan
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness',
                    'Insulin','BMI']] = diabetes_data_copy[['Glucose',
                                                            'BloodPressure',
                                                            'SkinThickness',
                                                            'Insulin',
                                                            'BMI']].replace(0,np.NaN)
# mostrando la suma de los datos vacíos
print(diabetes_data_copy.isnull().sum())
# mostrando histogramas correspondientes a cada columna
p = diabetes_data.hist(figsize = (20,20))
#%% Reemplazando los valores Nan de columnas con los valores promedios de las columnas o la mediana
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
# graficando los datos de las columnas sin NaN
p = diabetes_data_copy.hist(figsize = (20,20))
#%% Graficando la cantidad de columnas con tipos de datos int64 y float64
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
#%% Graficando diagrama de barras para identificar datos faltantes
import missingno as msno
pp=msno.bar(diabetes_data_copy)
# contando la cantidad de diabéticos y no diabéticos.
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
# graficando la cantidad de etiquetas 0 y 1
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
#%% Calculando matriz de correlación y realizando mapa de calor a los datos originales
plt.figure(figsize=(12,10)) 
# no se tiene en cuenta los valores nan
p=sns.heatmap(diabetes_data.dropna().corr(), annot=True,cmap ='RdYlGn')
plt.tight_layout()
#%% Calculando matriz de correlación y realizando mapa de calor al dataframe copy
plt.figure(figsize=(12,10))  
p=sns.heatmap(diabetes_data_copy.dropna().corr(), annot=True,cmap ='RdYlGn')
plt.tight_layout()
#%% estandarizando datos utilizando standar scaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# estandarizando las variables independientes
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
# seleccionando variable dependiente.
y = diabetes_data_copy.Outcome
#%% División en entrenamiento y prueba 
from sklearn.model_selection import train_test_split
"""
Cuando se usa stratify=y, el train_test_split intenta dividir los datos de tal 
forma que la proporción de clases en el conjunto de entrenamiento y el conjunto
de prueba sea la misma que en el conjunto original. Por ejemplo, si en el 
conjunto original y hay un 70% de una clase y un 30% de otra clase, stratify=y 
garantizará que estas proporciones se mantengan aproximadamente iguales tanto 
en y_train como en y_test.
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
#%% creando el modelo knn
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
# evaluando el desempeño de knn con diferentes vecinos
for i in range(1,15):

    knn = KNeighborsClassifier(i)
    #Entrenamiento del modelo KNN con los datos de entrenamiento:
    knn.fit(X_train,y_train)
    #Cálculo y almacenamiento de la puntuación del modelo en el conjunto de entrenamiento
    train_scores.append(knn.score(X_train,y_train))
    #Cálculo y almacenamiento de la puntuación del modelo en el conjunto de prueba:
    test_scores.append(knn.score(X_test,y_test))
# encontrando puntaje máximo de entrenamiento
max_train_score = max(train_scores)
# identificando los índices en la lista train_score donde se alcanza este puntaje máximo
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

# hacemos lo mismo para el test_score
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
#%% Graficando puntaje score obtenido en el entrenamiento y puntaje
plt.figure(figsize=(12,5))
p = sns.lineplot(x=range(1,15), y=train_scores, marker='*', label='Train Score')
p = sns.lineplot(x=range(1,15),y=test_scores,marker='o',label='Test Score')
"""
Podemos observar que el mejor resultado es en k=11
"""
# entrenando el modelo con k=11
knn = KNeighborsClassifier(11)
#entrenando modelo
knn.fit(X_train,y_train)
# evaluando el modelo
knn.score(X_test,y_test)
#%% visualización de regiones de decisión
from mlxtend.plotting import plot_decision_regions
"""
Visualizando como knn divide el espacio de caracteristicas en regiones de 
desición.Como plot_decision_regions solo puede visualizar datos en dos 
dimensiones, las demás características se fijan a valores constantes (20000) 
y se les asigna un rango de variación (20000) para simular un entorno realista. 
De este modo, puedes ver cómo el clasificador actúa en el espacio bidimensional 
mientras las otras dimensiones permanecen constantes.
"""
value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2, 
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=X_test.values)
plt.title('KNN with Diabetes Data')
plt.show()
#%% Procedemos a aplicarle la matriz de confusión
from sklearn.metrics import confusion_matrix
# obteniendo las predicciones del modelo 
y_pred = knn.predict(X_test)
# obteniendo matriz de confusión
confusion_matrix(y_test,y_pred)
# calculando matriz de confusión usando pandas
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#%% Visualizando matriz de confusión
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# obteniendo reporte
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#%% Curva ROC
from sklearn.metrics import roc_curve
# procedemos a predecir las probabilidades
y_pred_proba = knn.predict_proba(X_test)[:,1]
# obteniendo curva roc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# graficando curva roc
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=11) ROC curve')
plt.show()
#%% obteniendo el área bajo la curva
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)
#%% optimización de hiperparámetros
from sklearn.model_selection import GridSearchCV
# definiendo parámetros del modelo en un diccionario
param_grid = {'n_neighbors':np.arange(1,50)}
# creando modelo de clasificación knn
knn = KNeighborsClassifier()
# realizando búsqueda por malla
knn_cv= GridSearchCV(knn,param_grid,cv=5)
# ajustamos el modelo
knn_cv.fit(X,y)
# obteniendo la mejor puntuación y los mejores parámetros
print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))