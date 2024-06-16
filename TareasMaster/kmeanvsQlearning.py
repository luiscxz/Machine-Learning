# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:06:13 2024

@author: Luis A. García
"""

# importando librerías necesarias
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# procedemos a cargar los datos
iris = load_iris()
# seleccionando variables independientes
feature = pd.DataFrame(iris.data,
                       columns=iris.feature_names)
#Estandarización de los datos
from sklearn.preprocessing import StandardScaler
# definiendo escalador estándar
scaler = StandardScaler()
# escalando los datos
iris_scaled = scaler.fit_transform(feature)
# convertir datos escalados a dataframe
iris_scaled = pd.DataFrame(iris_scaled,
                           columns=feature.columns,
                           index = feature.index)
# Creación y entrenamiento del modelo k-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(iris_scaled)
# generando gráfica
plt.figure(figsize=(8,6))
plt.scatter(iris_scaled['sepal length (cm)'], iris_scaled['sepal width (cm)'], c=kmeans.labels_, cmap='viridis',
marker='o', edgecolor='k')
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('Clustering Iris Dataset with K-Means')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (standardized)')
plt.show()
# Procedemos a hacer predicciones
nueva_muestra = pd.DataFrame(
    {'sepal length (cm)': [5.1],
     'sepal width (cm)': [3.5],
     'petal length (cm)': [1.4],
     'petal width (cm)': [0.2]}
    )
# procedemos a estandarizar la muestra
nueva_muestra_scaled = scaler.transform(nueva_muestra)
nueva_muestra_scaled  = pd.DataFrame(nueva_muestra_scaled ,
                                     columns=nueva_muestra.columns)
# haciendo predicción
cluster_pred = kmeans.predict(nueva_muestra_scaled)
print(f"La muestra pertenece al cluster: {cluster_pred[0]}")
#%% Aprendizaje por refuerzo
import gym #para el entorno CartPole
# creando entorno Cartpole
env = gym.make('CartPole-v1')
""" estableciendo parámetros del algoritmo
    LEARNING_RATE: Tasa de aprendizaje
    DISCOUNT: Factor de descuento
    EPISODES: Número de episodios de entrenamiento que se usarán
    SHOW_EVERY: Frecuencia con la que se mostrará el entorno durante el entrenamiento.
"""
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500
# estableciendo tasa de exploración inicial
epsilon = 1 
# estableciendo episodio desde el cual se empieza a reducir el valor de épsilon
START_EPSILON_DECAYING = 1
# estableciendo episodio donde se detendrá la reducción de épsilon
END_EPSILON_DECAYING = EPISODES // 2
# calculando la cantidad por la cual se reducirá epsilon en cada épisodio.
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING -START_EPSILON_DECAYING)
#%%
"""
Creamos una tabla Q para almacenar los valores Q para cada par estado-acción
"""
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# calculando tamaño de cada intervalo en el espacio de observación discretizado del entorno CartPole-v1 de gym
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
"""
crea e inicializa una tabla Q (q_table) con valores aleatorios en el rango 
de -2 a 0. La tabla tiene dimensiones basadas en el espacio de observación 
discretizado y el espacio de acciones del entorno
"""
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#%% Definiendo función de discretización
"""
Debido a que el espacio de estados es continuo, necesitamos una función para
convertir los estados continuos a discretos
"""
def get_discrete_state(state):
    discrete_state = (state[0]-env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))
# Inicializando una lista para guardar las recompensas por episodio
ep_rewards = []

for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        observation, reward, done, _, _ = env.step(action)
        new_discrete_state = get_discrete_state(observation)
        
        episode_reward += reward
        
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action,)]
        
        if not done:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        else:
            new_q = reward
    
        q_table[discrete_state + (action,)] = new_q
        discrete_state = new_discrete_state
    
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)

env.close()

# Graficando las recompensas
plt.plot(range(EPISODES), ep_rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa')
plt.title('Recompensa por Episodio en el entorno CartPole')
plt.show()
