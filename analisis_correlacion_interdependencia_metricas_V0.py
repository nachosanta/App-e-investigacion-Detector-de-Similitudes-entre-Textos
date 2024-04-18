# ANALISIS DE CORRELACION LINEAL E INTERDEPENDENCIA ENTRE METRICAS CALCULADAS

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

os.chdir("C:\\Users\HP\Desktop\Tesis")

datos_1_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx')
datos_2_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx')
datos_3_df = pd.read_excel('Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx')

# concateno los df para tener todos los resultados de las métricas
metricas_df = pd.concat([datos_1_df, datos_2_df, datos_3_df], axis=0, ignore_index=True)


# cambio el nombre de algunas columnas
metricas_df = metricas_df.rename(columns={'Similitud_Damerau_Levenshtein_DIRECTO_palabras': 'Similitud_Damerau_Levenshtein', 'Similitud_Damerau_Levenshtein_DIRECTO_palabras_SinTransposicion': 'Similitud_Damerau_Levenshtein_SinTransposicion'})

# quito la palabra Similitud de los nombres de columnas
metricas_df.columns = metricas_df.columns.str.replace("Similitud_", "")

############################################################### grafico boxplots para ver las distribuciones comparativamente

# Configuración de estilo de seaborn (opcional)
sns.set(style="whitegrid")

# Crear un boxplot comparativo
plt.figure(figsize=(12, 8))
sns.boxplot(data=metricas_df.iloc[:, 1:], palette="Set3")
plt.title("Boxplots Comparativos de Métricas de Similitud entre los Informes Académicos")
plt.xticks(rotation=45, ha="right")  # Rotar etiquetas del eje x para mayor claridad
plt.show()


############################################### matriz de correlaciones lineales de Pearson y heatmap
metricas_df.columns

# CORRELACION GLOBAL
correlacion = metricas_df.iloc[:, 1:].corr().round(2)
annot_font_size = 6
annot_font_size_nombres = 8

# Ajustar la configuración global de DPI
plt.rcParams['figure.dpi'] = 300

# Crear el heatmap con ajustes de tamaño de letra
sns.heatmap(correlacion, cmap="coolwarm", annot=True, annot_kws={"size": annot_font_size},
            xticklabels=correlacion.columns, yticklabels=correlacion.columns, cbar=False)

# Ajustar el tamaño de la letra de los nombres de filas y columnas
plt.xticks(fontsize=annot_font_size_nombres)
plt.yticks(fontsize=annot_font_size_nombres)
plt.title("Heatmap de Correlación Lineal entre Similitudes Calculadas", fontsize=12)

plt.savefig("heatmap_global.png", dpi=300)

# Mostrar el gráfico
plt.show()

# calculo el determinante de la matriz para analizar multicolinealidad
det_corr = np.linalg.det(metricas_df.iloc[:, 1:].corr())
print(f'Determinante de la matriz de correlación: {det_corr}')


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

# GLOBAL
#sns.set(style='white', font_scale=1.6)
#g_global = sns.PairGrid(metricas_df.iloc[:, 1:], aspect=1.4, diag_sharey=False)
#g_global.map_lower(sns.regplot, lowess=False, ci=False, line_kws={'color': 'black'})
#g_global.map_diag(sns.histplot, bins=30)
#g_global.map_upper(corrdot)
#g_global.fig.suptitle("Correlación y Distribuciones de todas las Similitudes analizadas", y=1.02)



# Jacccard
metricas_jaccard_df = metricas_df[['Jaccard_palabras', 'Jaccard_3gramas', 'Jaccard_5gramas',
       'Jaccard_7gramas', 'Jaccard_9gramas', 'Jaccard_oraciones']]

sns.set(style='white', font_scale=1.6)
g_jacc = sns.PairGrid(metricas_jaccard_df, aspect=1.4, diag_sharey=False)
g_jacc.map_lower(sns.regplot, lowess=False, ci=False, line_kws={'color': 'black'})
g_jacc.map_diag(sns.histplot, bins=30)
g_jacc.map_upper(corrdot)
g_jacc.fig.suptitle("Distribuciones y Correlación entre Similitudes de Jaccard según distintas tokenizaciones", y=1.02)

# Coseno Count Vectorizer
metricas_coseno_df = metricas_df[['Coseno_CountVec_palabras', 'Coseno_CountVec_3gramas',
                                  'Coseno_CountVec_5gramas', 'Coseno_CountVec_7gramas',
                                  'Coseno_CountVec_9gramas']]

sns.set(style='white', font_scale=1.6)
g_cos = sns.PairGrid(metricas_coseno_df, aspect=1.4, diag_sharey=False)
g_cos.map_lower(sns.regplot, lowess=False, ci=False, line_kws={'color': 'black'})
g_cos.map_diag(sns.histplot, bins=30)
g_cos.map_upper(corrdot)
g_cos.fig.suptitle("Distribuciones y Correlación entre Similitudes de Coseno con Count Vectorizer según distintas tokenizaciones", y=1.02)

# Coseno tf-idf Vectorizer
metricas_coseno_tf_df = metricas_df[['Coseno_TFIDFVec_palabras',
                                  'Coseno_TFIDFVec_3gramas', 'Coseno_TFIDFVec_5gramas',
                                  'Coseno_TFIDFVec_7gramas', 'Coseno_TFIDFVec_9gramas']]

sns.set(style='white', font_scale=1.6)
g_cos_tf = sns.PairGrid(metricas_coseno_tf_df, aspect=1.4, diag_sharey=False)
g_cos_tf.map_lower(sns.regplot, lowess=False, ci=False, line_kws={'color': 'black'})
g_cos_tf.map_diag(sns.histplot, bins=30)
g_cos_tf.map_upper(corrdot)
g_cos_tf.fig.suptitle("Distribuciones y Correlación entre Similitudes de Coseno TF-IDF Vectorizer según distintas tokenizaciones", y=1.02)


# dam-lev y lcss
metricas_lev_lccss_df = metricas_df[['Damerau_Levenshtein', 
                                  'Damerau_Levenshtein_SinTransposicion', 
                                  'LCSSecquence', 'LCSString']]

sns.set(style='white', font_scale=1.6)
g_lev_lcss = sns.PairGrid(metricas_lev_lccss_df, aspect=1.4, diag_sharey=False)
g_lev_lcss.map_lower(sns.regplot, lowess=False, ci=False, line_kws={'color': 'black'})
g_lev_lcss.map_diag(sns.histplot, bins=30)
g_lev_lcss.map_upper(corrdot)
g_lev_lcss.fig.suptitle("Distribuciones y Correlación entre Similitudes de Damerau-Levenshtein y LCSS según variantes", y=1.02)


#################################################### analisis de interdependencia mediante Principal Components Analysis
from sklearn.decomposition import PCA
from sklearn import preprocessing

# estandarizo los datos restando su promedio y dividiendo por su desvio estandar (pasando a ser la media 0 y el desvio 1 en cada metrica)
metricas_df_stand = preprocessing.scale(metricas_df.iloc[:, 1:].T)

# entreno el PCA
pca = PCA()
pca_resultados = pca.fit_transform(metricas_df_stand)
# otra forma m que parece ser mas optimizada es hace directamente StandardScaler().fit_transform(metricas_df.iloc[:, 1:].T)

porcentaje_explicado_autovalores = np.round(pca.explained_variance_ratio_ * 100, decimals=2)


CP_labels = [str(x) for x in range(1, len(porcentaje_explicado_autovalores)+1)]


# Ajustar la configuración global de DPI
plt.rcParams['figure.dpi'] = 300


# grafico el porcentaje de varianza explicada por cada componente principal
plt.bar(x=range(1, len(porcentaje_explicado_autovalores)+1), height=porcentaje_explicado_autovalores, tick_label=CP_labels)
plt.xlabel('Componente Principal (CP)')
plt.ylabel('% de Varianza Explicada por esta CP')
plt.title('Información brindada por cada Componente Principal')

# Marcar los valores del eje y solo en las primeras 4 columnas
for i, label in enumerate(CP_labels[:4]):
    if i>=2:
        plt.text(i + 1, porcentaje_explicado_autovalores[i] + 1, f'{porcentaje_explicado_autovalores[i]:.1f}%',
                 ha='center', va='bottom', color=(96/255, 96/255, 96/255), fontsize=4.8)  # RGB (128, 128, 128)
    else:
        plt.text(i + 1, porcentaje_explicado_autovalores[i] + 1, f'{porcentaje_explicado_autovalores[i]:.1f}%',
                 ha='center', va='bottom', color=(96/255, 96/255, 96/255), fontsize=6.5)  # RGB (128, 128, 128)

# Aumentar el rango del eje y
plt.ylim(0, max(porcentaje_explicado_autovalores) + 5)  # Puedes ajustar el valor 5 según tus necesidades

# Add percentage symbols to the y-axis ticks
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])

plt.savefig("PCA.png", dpi=300)

plt.show()



''' NO HACE FALTA ESTE OTRO GRAFICO PORQUE YA SE VE CON EL OTRO
# Ajustar la configuración global de DPI
plt.rcParams['figure.dpi'] = 300

# Line plot of the cumulative explained variance
plt.plot(CP_labels, np.cumsum(porcentaje_explicado_autovalores), 'o-', markersize=5)
plt.xlabel('Componente Principal (CP)')
plt.ylabel('% de Varianza Explicada Acumulada por estas CP')
plt.title('Información Acumulada por Componentes Principales')

# Add percentage symbols to the y-axis ticks
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])

# Decrease the size of x-axis labels
#plt.xticks(fontsize=6)

# Add a red vertical line at CP3 & CP5
plt.axvline(x='1', color='red', linestyle='--', linewidth=1)
plt.axvline(x='3', color='black', linestyle=':', linewidth=1)

# Annotate the y value at CP3 & CP5 to the left of the point
y_value_at_CP1 = np.cumsum(porcentaje_explicado_autovalores)[0]  # Index 0 corresponds to CP1
plt.annotate(f'{y_value_at_CP1:.0f}%', xy=('1', y_value_at_CP1), xytext=(24, 0),
             textcoords='offset points', color='red', fontsize=8, ha='right')

y_value_at_CP3 = np.cumsum(porcentaje_explicado_autovalores)[4]  # Index 4 corresponds to CP3
plt.annotate(f'{y_value_at_CP3:.1f}%', xy=('3', y_value_at_CP3), xytext=(26, 0),
             textcoords='offset points', color='black', fontsize=8, ha='right')


plt.savefig("PCA_acumulado.png", dpi=300)

plt.show()
'''


from scipy.stats import pearsonr

# los autovectores (componentes principales)
autovectores_componentes = pca.components_.T # transpongo para que cada CP este en cada columna

autovectores_componentes_df = pd.DataFrame(autovectores_componentes)

autovectores_componentes_df.columns = ["CP " + label for label in CP_labels]

metricas_df_stand_df = pd.DataFrame(metricas_df_stand.T)

metricas_df_stand_df.columns = metricas_df.iloc[:, 1:].columns

###### analizo cuales son las variables originales que tuvieron mayor influencia sobre las primeras 5 componentes principales
'''# CP1 # esto no se si lo haría porque es mas confuso que ver la correlacion entre autovectores y las variables originales
autovec_cp1 = pd.Series(autovectores[0], index=metricas_df.iloc[:, 1:].columns)
autovec_cp1_ordenados = autovec_cp1.abs().sort_values(ascending=False)
top_10_memtricas_cp1 = autovec_cp1_ordenados[0:10].index.values
'''
###### Heat Map para correlacion de componentes principales (autovectores) versus las variables originales


# Seleccionar la columna "CP 1" del DataFrame autovectores_componentes_df
cp1_column = autovectores_componentes_df['CP 1']

# Calcular el coeficiente de correlación de Pearson entre "CP 1" y cada columna de metricas_df_stand_df
correlations_cp1 = metricas_df_stand_df.apply(lambda col: pearsonr(cp1_column, col)[0])

# Crear un DataFrame con los resultados
correlation_df_cp1 = pd.DataFrame({'Correlación con CP 1': correlations_cp1})

# Seleccionar la columna "CP 2" del DataFrame autovectores_componentes_df
cp2_column = autovectores_componentes_df['CP 2']

# Calcular el coeficiente de correlación de Pearson entre "CP 2" y cada columna de metricas_df_stand_df
correlations_cp2 = metricas_df_stand_df.apply(lambda col: pearsonr(cp2_column, col)[0])

# Crear un DataFrame con los resultados
correlation_df_cp2 = pd.DataFrame({'Correlación con CP 2': correlations_cp2})

# Seleccionar la columna "CP 3" del DataFrame autovectores_componentes_df
cp3_column = autovectores_componentes_df['CP 3']

# Calcular el coeficiente de correlación de Pearson entre "CP 3" y cada columna de metricas_df_stand_df
correlations_cp3 = metricas_df_stand_df.apply(lambda col: pearsonr(cp3_column, col)[0])

# Crear un DataFrame con los resultados
correlation_df_cp3 = pd.DataFrame({'Correlación con CP 3': correlations_cp3})

# Seleccionar la columna "CP 4" del DataFrame autovectores_componentes_df
cp4_column = autovectores_componentes_df['CP 4']

# Calcular el coeficiente de correlación de Pearson entre "CP 4" y cada columna de metricas_df_stand_df
correlations_cp4 = metricas_df_stand_df.apply(lambda col: pearsonr(cp4_column, col)[0])

# Crear un DataFrame con los resultados
correlation_df_cp4 = pd.DataFrame({'Correlación con CP 4': correlations_cp4})


correlation_PCA_df = pd.merge(pd.merge(correlation_df_cp1,
                                       correlation_df_cp2,
                                       left_index=True,
                                       right_index=True),
                              pd.merge(correlation_df_cp3,
                                       correlation_df_cp4,
                                       left_index=True,
                                       right_index=True),
                              left_index=True,
                              right_index=True)


# Ajustar la configuración global de DPI
plt.rcParams['figure.dpi'] = 300

# Visualizar la matriz de correlación
plt.figure(figsize=(4, 6))
sns.heatmap(correlation_PCA_df, cmap="coolwarm", annot=True, fmt=".2f", xticklabels=['CP 1', 'CP 2', 'CP 3', 'CP 4'], cbar=False)
plt.title("Correlación entre Componentes Principales (autovectores) y Variables Originales (métricas)")
plt.ylabel("")
plt.xlabel("")

# Ajustar la posición de las etiquetas del eje x en la parte superior
plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

plt.savefig("PCA_correlacion.png", dpi=300)

plt.show()