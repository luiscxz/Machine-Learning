# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:57:26 2024

Aprendizaje supervisado y no supervizado
https://www.kaggle.com/code/kanncaa1/machine-learning-tutorial-for-beginners/notebook

@author: Luis A. García
"""
# importando librerias necesarias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# accediendo a la ruta donde está el archivo csv
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\actividades\\modulo 5.3 act 1')
# leyendo archivo "column_2C_weka.csv"
data = pd.read_csv('column_2C_weka.csv')
# mostrando la lista de estilos de gráficos disponibles en Matplotlib
print(plt.style.available)
# seleccionando estilo ggplot
plt.style.use('ggplot')
#%% exploración de datos
# obteniendo información del dataframe (columnas, cantidad de datos no nullos y tipo de dato)
data.info()
# obteniendo resumen estadístico del dataframe
resumen = data.describe()
#%% Graficando las clases y analizando valanceo de las clases
"""
creando lista por comprensión que asigna color rojo si la clase es Abnormal
y asigna color verde si la clase es Normal
"""
color_list =['red' if i == 'Abnormal' else 'green' for i in data.loc[:,'class']]
# graficando todas las columnas, exceto la columna 'class'
scatter_matrix = pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [20,20],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 100,
                                       marker = '*',
                                       edgecolors= "black")
plt.show()
# rotando etiquetas
for ax in scatter_matrix.ravel():
    ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right')
#%% procedemos a ver que tan valanceadas están clases
grupo = data.groupby(by =['class'], dropna = False).agg(
    cantidad = ('class','count'),
    procentaje = ('class',lambda x: (len(x)/len(data))*100)
    ).reset_index()
# graficando la cantidad de clases
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
""" Las clases estan desbalanceadas:
    abnormal = 67.74%
    normal = 32.25%
"""
#%% implementando modelo knn sin optmización de hiperparámetros
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors = 3)
# x : variables independientes
# y : variable objetivo
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
# dividiendo en entrenamiento y prueba
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
# ajustando el modelo
knn.fit(x_train,y_train)
# obteniendo las predicciones
prediction = knn.predict(x_test)
# obteniendo precisión del modelo 
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) 
#%% Procedemos a buscar el valor de k que mayor precisión da
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
#%%
""" Utilizando solo dos características para la regresión:
    feature = pelvic_incidence
    target = sacral_slope
    En problemas de regresión e valor objetivo es una variable que varía continuamente
"""
# Seleccionando solo las filas con donde class == Abnormal
data1 = data[data['class']=='Abnormal']
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
#%% Implementando regresión lineal
from sklearn.linear_model import LinearRegression
# definiendo modelo
reg = LinearRegression()
# creando array que va desde el valor mínimo de x  hasta el máximo de x
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
# entrenando el modelo 
reg.fit(x,y)
# obteniendo predicciones para el array calculado predict_space
predicted = reg.predict(predict_space)
# mostrando coeficiente de determinación R2, El R² score mide qué tan bien el modelo de regresión se ajusta a los datos.
print('R^2 score: ',reg.score(x, y))
# graficando línea de regresión y scatter con datos originales
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
#%% creando modelo de regresión lineal que incluye penalización
# esto evita que el modelo se ajuste demasiado a los datos de entrenamiento
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# dividiendo datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.3)
# creando proceso que normaliza los datos y luego entrena el modelo de regresión Ridge
ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.1))
ridge.fit(x_train, y_train)
# procedemos a realizar predicciones
ridge_predict = ridge.predict(x_test)
# calculando puntuación score
print('Ridge score: ', ridge.score(x_test, y_test))
#%% Creando modelo de regresión lineal lasso incluye penalización de tipo L1
#(Para reducir algunos coeficientes a cero)
from sklearn.linear_model import Lasso
# Seleccionando las variables dependientes
x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])
# dividiendo en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.3)
# Crear pipeline que normaliza los datos y luego entrena el modelo Lasso
lasso_pipeline = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1))
lasso_pipeline.fit(x_train, y_train)

# Realizar predicciones y evaluar el modelo
lasso_predict = lasso_pipeline.predict(x_test)
print('Lasso score: ', lasso_pipeline.score(x_test, y_test))

# Acceder al modelo Lasso y al StandardScaler dentro del pipeline
scaler = lasso_pipeline.named_steps['standardscaler']
lasso_model = lasso_pipeline.named_steps['lasso']

# Ajustar los coeficientes para aplicarlos a las características no escaladas
adjusted_coefs = lasso_model.coef_ / scaler.scale_
print('Lasso coefficients: ', adjusted_coefs)
#%% Procedemos a usar la matriz de confución con random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# seleccionando los datos dependientes e independientes
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
# separando en entrenamiento y prueba
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
# creando clasificador de bosques aleatorios con estado aleatorio fijo = 4
rf = RandomForestClassifier(random_state = 4)
# entrenando el modelo y obteniendo las predicciones
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
# calculando matriz de confusión
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
"""
Confusion matrix: 
 [[60  6]
 [ 8 19]]
 Esto significa que el modelo tiene 60 verdaderos negativos, 6 falso positivo,
 8 falsos negativos y 19 verdaderos positivos
"""
print('Classification report: \n',classification_report(y_test,y_pred))
# Visualizando  matriz de confusión
plt.figure(figsize=(10, 12))  
heatmap = sns.heatmap(
    cm,  
    cmap='inferno', 
    annot=True, 
    fmt='.2f',
    cbar=True,
    square=True,
    annot_kws={'size': 12, 'fontweight': 'bold', 'fontfamily': 'Arial'},  
    linewidth=.5,
    linecolor='none'
)
# Ajustar las etiquetas del eje x para que sean más legibles
plt.xticks(fontsize=12, fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=12, fontweight='bold', fontfamily='Arial')
#nombres de ejes
plt.title('Matriz de confusión', fontname='Arial', fontweight='bold',fontsize=17)

#%% Curva ROC
"""
ROC es la curva característica de operación del receptor.
En esta curva, el eje x es la tasa de falsos positivos y el eje y es la tasa de
verdaderos positivos. Si la curva en el gráfico está más cerca de la esquina 
superior izquierda, la prueba es más precisa.

La puntuación de la curva ROC es el AUC, que es el área de computación bajo la 
curva a partir de las puntuaciones de predicción. Queremos que el AUC esté más 
cerca de 1
"""
# curva ROC con regresión logistica
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# creando columna de data frame donde 1 es Abnormal y 0 es Normal
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
# seleccionando variables dependientes e independientes
x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
# separando en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
# creando modelo de regresión logística y entrenándolo
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
# obteniendo predicciones en términos de probabilidad
y_pred_prob = logreg.predict_proba(x_test)[:,1]
"""
Calculando curva ROC:
    fpr: Es una variable que almacenará la tasa de falsos positivos
    tpr: Es una variable que almacenará la tasa de verdaderos positivos 
    thresholds: Es una variable que almacenará los umbrales de decisión 
    utilizados para calcular las tasas de falsos positivos y verdaderos positivos
"""
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# graficando curva ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
#%% Aplicación de validación cruzada en KNN
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x,y)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
#%% Aplicando validación cruzada a regresión logística 
"""
Otro ejemplo de búsqueda en cuadrícula con 2 hiperparámetros

El primer hiperparámetro es C: parámetro de regularización de regresión logística
Si C es alto: sobreajuste
Si C es bajo: infraajuste
El segundo hiperparámetro es penalización (función de pérdida): l1 (Lasso) o
 l2 (Ridge) como aprendimos en la parte de regresión lineal.
"""
# estableciendo diccionario de parámetros
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
# dividiendo en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)
# creando modelo y haciendo búsqueda por malla
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
# ajustando el modelo
logreg_cv.fit(x_train,y_train)
# mostrando mejores parámetros y puntuación accuracy
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))
#%% Ejemplo de preprocesamiento
# leyendo los datos
data = pd.read_csv('column_2C_weka.csv')
# codificando variables categóricas a dummy
df = pd.get_dummies(data,dtype=int)
# Eliminando la columna Class_Normal
df.drop("class_Normal",axis = 1, inplace = True) 
df.head(10)
#%%  Uso de máquina de soporte vectorial y
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
"""creando lista que contiene los pasos que seguirá la pipeline.
1) Estandariza los datos
2) entrena el modelo svc
"""
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
# creando diccionario que contiene los parámetros a usar en SVM
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
# dividiendo en entrenamiento en prueba
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
# Realizando búsqueda por malla
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
# entrenando el modelo
cv.fit(x_train,y_train)
# obteniendo predicciones
y_pred = cv.predict(x_test)
# Obteniendo puntuación Acurracy y mejores parámetros del modelo
print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
#%% Aprendizaje no supervisado
"""
En esta sección eliminaremos la columna objetivo
Utilizaremos el método de clusterización K-means
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# accediendo a la ruta donde está el archivo csv
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\actividades\\modulo 5.3 act 1')
# leyendo archivo "column_2C_weka.csv"
data = pd.read_csv('column_2C_weka.csv')
# graficando relación entre pelvic_radius y degree_spondylolisthesis
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
#%% creando modelo de clusterización Kmeans con dos clústeres
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
# entrenando el modelo
kmeans.fit(data2)
# obteniendo las etiquetas predichas por el modelo
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# procedemos a evaluar el agrupamiento usando una tabla de tabulación cruzada
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
#%% Realizando análisis de inercia para determinar el número óptimo de clústeres
# creando array vacío con capacidad de 8 elementos
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()
"""
este código ayuda a identificar visualmente el codo 
(el punto donde la tasa de disminución de la inercia disminuye significativamente) 
en el gráfico de inercia, lo que puede sugerir el número óptimo de clústeres 
para usar en el algoritmo KMeans en función de la estructura de los datos.
"""
#%% Procedemos a estandarizar los datos
data3 = data.drop('class',axis = 1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# creando escalador estándar
scalar = StandardScaler()
# creando modelo con 2 clústeres
kmeans = KMeans(n_clusters = 2)
# creando pipelinea 
pipe = make_pipeline(scalar,kmeans)
# entrenando modelo con datos estandarizados
pipe.fit(data3)
# obteniendo las etiquetas
labels = pipe.predict(data3)
# Guardando etiquetas en un dataframe
df = pd.DataFrame({'labels':labels,"class":data['class']})
# creando tabla de frecuencias cruzadas. lo que permite analizar la relación 
#entre las dos variables 'labels' y 'class' en términos de su distribución conjunta
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
#%% Agrupamiento jerárquico
from scipy.cluster.hierarchy import linkage,dendrogram
# realizando agrupamiento jerárquico por el método simple, usando las líneas 200:220
merg = linkage(data3.iloc[200:220,:],method = 'single')
# visualizando dendrograma
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()
#%% Reducción de dimensionalidad
from sklearn.manifold import TSNE
# creando modelo con tasa de aprendizaje de 100
model = TSNE(learning_rate=100)
color_list =['red' if i == 'Abnormal' else 'green' for i in data.loc[:,'class']]
# Ajustando el modelo
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]
# graficando
plt.scatter(x,y,c = color_list )
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
#%% Análisis de componentes principales
from sklearn.decomposition import PCA
# creando modelo 
model = PCA()
# entrenando modelo
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)
#%% Varianza por cada componente principal
# creando instancia que estandariza los datos
scaler = StandardScaler()
# creando modelo de componentes principales
pca = PCA()
# creando pipelinea
pipeline = make_pipeline(scaler,pca)
# entrenando el modelo
pipeline.fit(data3)
# graficando componentes principales y varianza de cada componente
plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()
#%% analizando las 2 comprimeras componentes prinicipales
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
# visualizando la relación entre las dos componentes principales
plt.scatter(x,y,c = color_list)
plt.show()