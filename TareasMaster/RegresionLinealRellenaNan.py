# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:45:45 2024
Regresión simple para rellenar espacios vacios
@author: Luis A. García
"""
# importando librerías necesarias
import os
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, geom_line
# accediendo a la ruta del archivo
os.chdir('D:\\6. NEXER\\master\\Contenido\\5. Machine Learning\\tareas entregadas\\modulo 5.4 act2')
#leyendo archivo
file = pd.read_csv('calidad-del-aire-datos-historicos-diarios.csv',sep=';')
# creando gráfico de dispersión
(
 ggplot(file, aes(x='PM25 (ug/m3)', y='NO2 (ug/m3)')) +
        geom_point(color='blue') +
        geom_smooth(method='lm', se=False, color='red') +
        labs(x="PM25 [mg/m^3]", y="NO2 [µg/m^3]", title="Gráfico de dispersión: PM25 vs NO2")
        )
#%% aplicación del modelo de regresión linea
import statsmodels.api as sm               
df = file[['PM25 (ug/m3)','NO2 (ug/m3)']]
# renombrando columnas del dataframe
df =df.rename(columns={'PM25 (ug/m3)':'PM25','NO2 (ug/m3)':'NO2'})
#creando y entrenando modelo de regresión lineal de Statsmodels
modelo = sm.formula.ols(formula='PM25 ~ NO2', data=df).fit()
# Mostrar el resumen del modelo
print(modelo.summary())
#%% procedemos a obtener los datos correspondientes a los primeros 14 días del mes de febrero
#del 2014 y correspondientes a la provincia de Valladolid
condicion = (file['Fecha']>='2014-02-01') & (file['Fecha']<='2014-02-14') & (file['Provincia']=='Valladolid')
ValladolidFeb2014 = file[condicion]
# convirtiendo fecha a formato datetime
ValladolidFeb2014['Fecha'] = pd.to_datetime(ValladolidFeb2014['Fecha']).dt.strftime('%Y-%m-%d')
# gráficando
(
    ggplot(ValladolidFeb2014, aes(x='Fecha', y='PM10 (ug/m3)')) +
    geom_point(color='black', fill='white', size=2.5, stroke=1) +
    geom_line(aes(x='Fecha', y='PM10 (ug/m3)'), color='red') +  # Agregar línea que conecta los puntos
    geom_smooth(method='lm', se=False, color='red') +
    labs(x='Fecha', y='PM10 [ug/m^3]', title='Datos PM10 en Valladolid (Feb 2014)')
)
# ajustando el modelo
ValladolidFeb2014 =ValladolidFeb2014.rename(columns={'PM10 (ug/m3)':'PM10'})
modeloPM10 = sm.OLS.from_formula('Q("PM10") ~ Fecha', data=ValladolidFeb2014)
resultado = modeloPM10.fit()

#%%


