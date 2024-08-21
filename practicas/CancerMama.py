# -*- coding: utf-8 -*-
"""
Algoritmo que usa diferentes modelos para detectar cancer de mama

@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt
# Accediendo a donde están los datos
os.chdir('D:\\3. Cursos\\9. Data machine learning\\CancerDeMama')
# Leyendo archivo csv
file = pd.read_csv('Cancer_Data.csv')
#eliminando columna vacía
file =file.drop('Unnamed: 32',axis=1)
#%% Explorando datos
# consultado nombre de columnas, tipo y cantidad de registros no-null
file.info()
# realizando análisis de balanceo de clases
tiposCancer = file.groupby(by = ['diagnosis']).agg(
    # calculando la cantidad de clases
    cantidad = ('diagnosis','count'),
    # calculando el porcentaje que ocupa cada clase
    porcentaje = ('diagnosis', lambda x: (len(x)/len(file))*100)
).reset_index()
""" Dado que:
    Registros de cancer B = 62,74%
    Registros de cancer M = 37.25%
    Se observa que las etiquetas estan desbalanceadas 
"""
# Grafincando las clases y sus porcentajes
fig, ax = plt.subplots(figsize=(8, 3)) 

# Creando gráfica de barras horizontal
bars = ax.barh(tiposCancer['diagnosis'], tiposCancer['cantidad'], color=['#00bfff', '#ff7f7f'],height=0.5)
# Añadir el texto en las barras
for bar, porcentaje in zip(bars, tiposCancer['porcentaje']):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
            f'{porcentaje:.2f}%', va='center', ha='left')

# Mejorar la apariencia general
ax.set_title("Distribución de Clases", fontsize=16, weight='bold')
ax.set_xlabel("Número de Casos", fontsize=12)
ax.set_ylabel("Clase", fontsize=12)
ax.set_xlim(0, 500)  # Ajuste de los límites en el eje x
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(range(len(tiposCancer['diagnosis'])))
ax.set_yticklabels(tiposCancer['diagnosis'], fontsize=12)

# Ajustar el espaciado entre las barras
plt.subplots_adjust(left=0.15, right=0.85, top=0.50, bottom=0.15)

#-- Realizando diagrama de cajas y bigotes 
columnas =file.columns.to_list()
fig = plt.subplots(figsize=(15, 3)) 
boxplotChl = file.boxplot(column= columnas[2:31],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Características', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('Valores', fontsize=20, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
_=boxplotChl.set_xticklabels(boxplotChl.get_xticklabels(), rotation=90, fontsize=12)
""" Con el diagrama de cajas se pudo observar los siguiente:
    area_mean: 0 hasta 2500
    area_se: 0 hasta 520
    area_worst : 0 a casi 5000
    Lo que indica que estas columnas generan más peso, entonces debemos
    probar normalizando o estandarizando 
"""
#%% Extracción de una muestra
from sklearn.model_selection import train_test_split
# definiendo tamaño de muestra igual al 1%
sample_size = 0.01
data, muestra = train_test_split(
    file, 
    test_size=sample_size, 
    stratify=file['diagnosis'], 
    random_state=42
)
# reseteando index para la muestra y lo datos que serán para entrenamiento y validación
muestra= muestra.reset_index(drop=True)
data = data.reset_index(drop=True)
#%% Normalización o estandarización de los datos
from sklearn import preprocessing
# Separando en variables dependientes e independientes
objetivo = data['diagnosis']
independientes = data.loc[:,~data.columns.isin({'id','diagnosis'})]
# conservando nombre de las columnas del df independientes
col_ind = independientes.columns
# creando escalador para normalizar los datos 
normalizador = preprocessing.MinMaxScaler()
# normalizando con MinMax: a cada columna le resta su respectivo minimo y divide entre su respectivo (max-min)
independientes = pd.DataFrame(normalizador.fit_transform(independientes),
                              columns=col_ind)
fig = plt.subplots(figsize=(15, 3)) 
boxplotChl = independientes.boxplot(column= col_ind.to_list(),
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Características', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('Valores normalizados', fontsize=18, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
_=boxplotChl.set_xticklabels(boxplotChl.get_xticklabels(), rotation=90, fontsize=7)
# Procedemos a codificar las variables Categoricas de forma manual
objetivo = objetivo.replace({'M':1,'B':0})
objetivo = objetivo.astype(int)
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

parametros_logistica = [
    {
        "penalty": ["l1"],
        "C": [0.1, 1, 10],
        "max_iter": [3000, 5000, 8000],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"]
    },
    {
        "penalty": ["l2"],
        "C": [0.1, 1, 10],
        "max_iter": [3000, 5000, 8000],
        "solver": ["liblinear", "newton-cg", "sag", "lbfgs"],
        "class_weight": [None, "balanced"]
    },
    {
        "penalty": ["elasticnet"],
        "C": [0.1, 1, 10],
        "max_iter": [3000, 5000, 8000],
        "solver": ["saga"],
        "class_weight": [None, "balanced"],
        "l1_ratio": [0.1, 0.5, 0.9]  # Requerido para elasticnet
    }
]

parametros_busqueda_svm = {
    "degree": [1,2,3,4],
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["poly", "rbf"]}

parametros_xgboost= {
    "max_depth": list(range(3, 7)),  
    "learning_rate": [0.05, 0.1],   
    "n_estimators": [100, 200],     
    "gamma": [0, 0.1],              
    "reg_alpha": [0, 0.1],       
    "reg_lambda": [1],              
}
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
        Devuelve las métricas calculadas.
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
import xgboost
# Definiendo modelos 
estimador_knn = KNeighborsClassifier()
estimador_svm = SVC()
estimador_arbol = tree.DecisionTreeClassifier()
estimador_arbol_aleatorio = tree.ExtraTreeClassifier()
estimador_log_reg = LogisticRegression()
estimador_xgboost = xgboost.XGBClassifier(eval_metric='logloss')
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

log_reg_Random = RandomizedSearchCV(estimator=estimador_log_reg, 
                    param_distributions=parametros_logistica,
                    scoring="f1_macro", n_jobs=-1, n_iter=4)

xgboost_Random = RandomizedSearchCV(estimator=estimador_xgboost, 
                    param_distributions=parametros_xgboost,
                    scoring="f1_macro", n_jobs=-1, n_iter=4)

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

log_reg_grid = GridSearchCV(estimator=estimador_log_reg, 
                    param_grid=parametros_logistica,
                    scoring="f1_macro", n_jobs=-1)

xgboost_grid = GridSearchCV(estimator=estimador_xgboost, 
                    param_grid=parametros_xgboost,
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
log_reg_Random.fit(independientes, objetivo)
xgboost_Random.fit(independientes, objetivo)
# ajustando modelos por busqueda en malla
knn_grid.fit(independientes, objetivo)
svm_grid.fit(independientes, objetivo)
arbol_grid.fit(independientes, objetivo)
log_reg_grid.fit(independientes, objetivo)
xgboost_grid.fit(independientes, objetivo)
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

resultados["log_reg_grid"] = evaluar_modelo(log_reg_grid.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["xgboost_grid"] = evaluar_modelo(xgboost_grid.best_estimator_,
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
resultados["log_reg_Random"] = evaluar_modelo(log_reg_Random.best_estimator_,
                                   independientes,
                                   objetivo)
resultados["xgboost_Random"] = evaluar_modelo(xgboost_Random.best_estimator_,
                                   independientes,
                                   objetivo)

resultados["arbol_aleatorio"] = evaluar_modelo(estimador_arbol_aleatorio,
                                               independientes,
                                               objetivo)
#%%# definiendo función que muestra los resultados 
def ver_resultados(resultados):
    # Convertir el diccionario de resultados en un DataFrame y transponerlo
    resultados_df = pd.DataFrame(resultados).T
    # Iterar sobre cada columna del DataFrame
    for col in resultados_df:
        """
        Cada fila de la columna contiene una lista de valores, por ejemplo:
        [0.00399971, 0.00400662, 0.00399971, 0.00399923, 0.00400662]
        Para simplificar, se calcula el promedio de los valores en cada fila
        y se reemplaza la lista con este valor promedio.
        """
        resultados_df[col] = resultados_df[col].apply(np.mean)
        
        """
        Normalizar los valores de la columna dividiendo cada valor por el máximo 
        valor en la columna. Esto permite escalar los datos entre 0 y 1, 
        facilitando la comparación entre modelos.
        """
        resultados_df[col + "_idx"] = resultados_df[col] / resultados_df[col].max()
    
    # Devolver el DataFrame procesado con los resultados y las columnas normalizadas
    return resultados_df
#%% observando los resultados
resultados_df= ver_resultados(resultados)
# organizando resultados.
resultados_df = resultados_df.sort_values(by=['test_score', 'fit_time'], ascending=[False, True])
""" una vez observado e identificado el modelo con mejor puntuación f1_score,
procedemos a identificar sus mejores parámetros
"""
log_reg_grid.best_params_
mejores_params = log_reg_grid.best_params_
#%% Volvemos a crear el modelo y esta vez lo corremos con los mejores parámetros
mejorModelo = LogisticRegression(**mejores_params)
# entrenando el modelo
mejorModelo.fit(independientes, objetivo)
#%% Evaluando el rendimiento final del modelo con la muestra
# separando en variables independietes y objetivo
MIndependiente = muestra.loc[:,~muestra.columns.isin({'id','diagnosis'})]
Mobjetivo = muestra['diagnosis']
# normalizando los datos con el escalor creado para normalizar
MIndependiente = pd.DataFrame(normalizador.transform(MIndependiente), columns=col_ind)
# codificando variables
Mobjetivo = Mobjetivo.replace({'M':1,'B':0})
# convirtiendo columna a tipo int
Mobjetivo = Mobjetivo.astype(int)

# Evalúa el modelo con el 1% de los datos separados y normalizados
y_pred = mejorModelo.predict(MIndependiente)
#%% Calcuando métricas de evaluación final
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
accuracy = accuracy_score(Mobjetivo, y_pred)
f1 = f1_score(Mobjetivo, y_pred, average='weighted')  # Cambia a 'macro' o 'micro' según sea necesario
conf_matrix = confusion_matrix(Mobjetivo, y_pred)
# obtenindo reporte
print(classification_report(Mobjetivo, y_pred))

#%% Procedemos a ver como influyen las características en las decisiones del modelo
from sklearn.inspection import permutation_importance
import seaborn as sns
result = permutation_importance(mejorModelo, independientes, objetivo, n_repeats=30, random_state=42, n_jobs=-1)
# Preparar los datos para el gráfico
sorted_idx = result.importances_mean.argsort()[::-1]  # Invertir el orden
importances = result.importances_mean[sorted_idx]
features = np.array(independientes.columns)[sorted_idx]
# graficando
plt.figure(figsize=(10, 6),dpi=200)
sns.barplot(x=importances, y=features, hue=features, palette='Set1', edgecolor='k', legend=False)
plt.xlim(-0.001, 0.045)
plt.xlabel("Importancia (permutación)")
plt.ylabel("Características")
plt.title("Importancia de las características")
# Añadir línea vertical en el eje X
plt.axvline(x=0, color='k', linestyle='--')
for index, value in enumerate(importances):
    plt.text(value + 0.001, index, f'{value:.3f}', va='center', fontsize=10)

# Mejorar la visualización
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
#%% gráfico radiaL
# Crear un gráfico radial
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.bar(angles, importances, width=0.4, align='edge')

ax.set_xticks(angles)
ax.set_xticklabels(features, fontsize=10)
ax.yaxis.set_tick_params(labelsize=10)

plt.title('Permutation Importance (Radial)', fontsize=15)
plt.show()
#%%
import plotly.express as px
# Crear un DataFrame para plotly
df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# Crear el gráfico de pie con estilo similar
fig = px.pie(df, values='Importance', names='Feature',
             title='Permutation Importance (Pie Chart)',
             hover_name='Feature',
             hole=0.5,  # Agujero central
             labels={'Feature': 'Feature', 'Importance': 'Importance'},
             template='plotly',  # Estilo plotly
            )

# Añadir anotaciones para mostrar las etiquetas dentro de cada segmento
fig.update_traces(textposition='inside', textinfo='percent+label')

# Mostrar el gráfico en el navegador
fig.show(renderer='browser')
#fig.show()
#%% Guardando modelo
import pickle
""" 
abriendo archivo modelo_pca.pickle en modo de escritura binara "wb" y 
Serializando el objeto modelo_pca y escribiendo en el archivo abierto.
"""
ruta = 'D:\\15. Notas de clase\\EjmplosML'
# Guardando los datos del modelo de normalización
with open(ruta+"\\Normalizador.pickle", "wb") as file:
    pickle.dump(normalizador , file)
# procedemos a guardar el mejor modelo
with open (ruta+"\\modeloRegresionLogistica.pickle", "wb") as file:
    pickle.dump(mejorModelo, file)
#%% Abriendo modelo
import os
import pickle
os.chdir('D:\\15. Notas de clase\\EjmplosML')
# leyendo modelo de normalización
with open('Normalizador.pickle','rb') as file:
    modeloPREPROCESAMIENTO = pickle.load(file)
# leyendo modelo de clasificación
with open('modeloRegresionLogistica.pickle','rb') as file:
    mejorMODELO = pickle.load(file)
#%%
import pandas as pd
os.chdir('D:\\3. Cursos\\9. Data machine learning\\CancerDeMama')
dfNuevo = pd.read_csv('cancemamadf.csv')
# separando en variables idependientes y objetivo
dfNuevoindependiente = dfNuevo.loc[:,~dfNuevo.columns.isin({'id','diagnosis','Unnamed: 32'})]
dfNuevoobjetivo = dfNuevo['diagnosis']
# transformando(normalizando) datos
nuevoDATA = pd.DataFrame(modeloPREPROCESAMIENTO.transform(dfNuevoindependiente),
                         columns=dfNuevoindependiente.columns.to_list())

# obteniendo prediciones del modelo 
predicciones=mejorMODELO.predict(nuevoDATA)
resultados_mapeados = ['M' if pred == 1 else 'B' for pred in predicciones]

print(resultados_mapeados)