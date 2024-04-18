# CLUSTERING

# hago un analisis de clustering para cada conjunto de informes por separado porque todos deben tener sus similitudes calculadas

import os
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt


os.chdir("C:\\Users\HP\Desktop\Tesis")

datos_1_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx')
datos_2_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx')
datos_3_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx')

# me quedo solo con las metricas necesarias
datos_1_df = datos_1_df[['ID_unico', 'Similitud_Jaccard_3gramas', 'Similitud_Jaccard_palabras', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']]
datos_2_df = datos_2_df[['ID_unico', 'Similitud_Jaccard_3gramas', 'Similitud_Jaccard_palabras', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']]
datos_3_df = datos_3_df[['ID_unico', 'Similitud_Jaccard_3gramas', 'Similitud_Jaccard_palabras', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']]


# funcion para convertir el df de pares con similitudes a una matriz de distacias
def df2distanceMatrix(df):
    # promedio ponderado a partir de la importancia de las variables segun el gradient boosting
    ponderaciones_importancia_gini = np.array([0.32014095, 0.16032293, 0.03191859, 0.48761753])
    df['Distancia'] = 1-(df[['Similitud_Jaccard_palabras', 'Similitud_Jaccard_3gramas', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']] * ponderaciones_importancia_gini).sum(axis=1)   
    #df['Distancia'] = 1-df[df.columns.difference(['ID_unico'])].mean(axis=1).astype(float)
    df['ID_1'] = df['ID_unico'].str.split('_').str[-2]
    df['ID_2'] = df['ID_unico'].str.split('_').str[-1]
    
    # Reordena los índices para asegurarte de que los mismos IDs estén en las mismas posiciones
    idx = sorted(set(df['ID_1'].unique()) | set(df['ID_2'].unique()))
    
    # Crea la tabla pivote asegurándote de que los índices coincidan
    pivot_table = df.pivot_table(index='ID_1', columns='ID_2', values='Distancia', fill_value=0)
    pivot_table = pivot_table.reindex(index=idx, columns=idx, fill_value=0)
    
    # Suma la tabla pivote con su traspuesta para hacerla simétrica
    dist_matrix = pivot_table.values + pivot_table.T.values
    
    # Obtén los nombres de las columnas para usarlos como nombres de las filas y columnas de la matriz
    nombres_columnas = pivot_table.columns
    
    # Crea un DataFrame con la matriz y los nombres de las columnas como índices
    dist_matrix = pd.DataFrame(dist_matrix, index=nombres_columnas, columns=nombres_columnas)

    
    return dist_matrix

df_matrix_1 = df2distanceMatrix(datos_1_df)
df_matrix_2 = df2distanceMatrix(datos_2_df)
df_matrix_3 = df2distanceMatrix(datos_3_df)

IDs_ordenados_matrix_1 = df_matrix_1.columns
IDs_ordenados_matrix_2 = df_matrix_2.columns
IDs_ordenados_matrix_3 = df_matrix_3.columns

dist_matrix_1 = df_matrix_1.values
dist_matrix_2 = df_matrix_2.values
dist_matrix_3 = df_matrix_3.values


# Realizar clustering jerárquico
# prueba distintos linkage


# WARD
Z1 = hierarchy.linkage(dist_matrix_1, method='ward')
Z2 = hierarchy.linkage(dist_matrix_2, method='ward')
Z3 = hierarchy.linkage(dist_matrix_3, method='ward')


# Representación gráfica del dendrograma
plt.figure(figsize=(18, 6), dpi=300)
dn = hierarchy.dendrogram(Z1)
plt.axhline(y=1.3, color='r', linestyle='--')
plt.title('Dataset 1: linkage Ward')
plt.xticks(rotation=45, fontsize=9)
plt.show()

plt.figure(figsize=(18, 6), dpi=300)
dn = hierarchy.dendrogram(Z2)
plt.axhline(y=1.3, color='r', linestyle='--')
plt.title('Dataset 2: linkage Ward')
plt.xticks(rotation=45, fontsize=9)
plt.show()

plt.figure(figsize=(18, 6), dpi=300)
dn = hierarchy.dendrogram(Z3)
plt.axhline(y=1.3, color='r', linestyle='--')
plt.title('Dataset 3: linkage Ward')
plt.xticks(rotation=45, fontsize=9)
plt.show()



'''
# SINGLE
Z1 = hierarchy.linkage(dist_matrix_1, method='single')
Z2 = hierarchy.linkage(dist_matrix_2, method='single')
Z3 = hierarchy.linkage(dist_matrix_3, method='single')

# Representación gráfica del dendrograma
plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z1)
plt.title('Dataset 1: linkage Single')
plt.show()

plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z2)
plt.title('Dataset 2: linkage Single')
plt.show()


plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z3)
plt.title('Dataset 3: linkage Single')
#plt.tick_params(axis='x', labelsize=4)
plt.show()

# CENTROID
Z1 = hierarchy.linkage(dist_matrix_1, method='centroid')
Z2 = hierarchy.linkage(dist_matrix_2, method='centroid')
Z3 = hierarchy.linkage(dist_matrix_3, method='centroid')

# Representación gráfica del dendrograma
plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z1)
plt.title('Dataset 1: linkage Centroid')
plt.show()

plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z2)
plt.title('Dataset 2: linkage Centroid')
plt.show()


plt.figure(dpi=100)
dn = hierarchy.dendrogram(Z3)
plt.title('Dataset 3: linkage Centroid')
#plt.tick_params(axis='x', labelsize=4)
plt.show()
'''

# CORTE / DISTANCIA MAXIMA

# Definir la distancia límite
t = 1.3

# Realizar el corte en la distancia límite
clusters_1 = hierarchy.fcluster(Z1, t, criterion='distance')
clusters_2 = hierarchy.fcluster(Z2, t, criterion='distance')
clusters_3 = hierarchy.fcluster(Z3, t, criterion='distance')

# armo un df con los IDs y los grupos
informes_clusters_1 = pd.concat([pd.DataFrame(list(IDs_ordenados_matrix_1), columns=['IDs']), pd.DataFrame(clusters_1, columns=['cluster'])], axis=1)
informes_clusters_2 = pd.concat([pd.DataFrame(list(IDs_ordenados_matrix_2), columns=['IDs']), pd.DataFrame(clusters_2, columns=['cluster'])], axis=1)
informes_clusters_3 = pd.concat([pd.DataFrame(list(IDs_ordenados_matrix_3), columns=['IDs']), pd.DataFrame(clusters_3, columns=['cluster'])], axis=1)

# Agrupar por 'cluster' y contar el número de observaciones en cada grupo para quedarme solo con grupos de 3 o más
grupos_con_observaciones_1 = informes_clusters_1.groupby('cluster').filter(lambda x: len(x) >= 3)
grupos_con_observaciones_2 = informes_clusters_2.groupby('cluster').filter(lambda x: len(x) >= 3)
grupos_con_observaciones_3 = informes_clusters_3.groupby('cluster').filter(lambda x: len(x) >= 3)

# Obtener los grupos con 2 o más observaciones y los IDs correspondientes
grupos_con_observaciones_ids_1 = grupos_con_observaciones_1.groupby('cluster')['IDs'].apply(list).to_frame()
grupos_con_observaciones_ids_2 = grupos_con_observaciones_2.groupby('cluster')['IDs'].apply(list).to_frame()
grupos_con_observaciones_ids_3 = grupos_con_observaciones_3.groupby('cluster')['IDs'].apply(list).to_frame()


# Función para convertir una lista de IDs en un string separado por comas
def ids_a_string(ids):
    return ', '.join(map(str, ids))

# Aplicar la función a la columna 'IDs' para convertir listas en strings
grupos_con_observaciones_ids_1['IDs_string'] = grupos_con_observaciones_ids_1['IDs'].apply(ids_a_string)

# Crear un nuevo DataFrame con las columnas deseadas
df_final_grupos = pd.DataFrame({
    'Grupos de informes a revisar': [f'Grupo {chr(65 + i)}' for i in range(len(grupos_con_observaciones_ids_1))],
    'IDs incluidos en este grupo': grupos_con_observaciones_ids_1['IDs_string']
})


