import os # para manejar el operative system
import re # regular expressions
#from io import StringIO
#from io import BytesIO # lo usamos para pasar a modo binario y poder leer directo en pdf
import pandas as pd
#import PyPDF2
#from PyPDF2 import PdfReader # lector de pdfs
import zipfile # para descomprimir los archivos en el zip
from unidecode import unidecode # para reemplazar tildes por las letras sin tildes
import nltk
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer # ESTE LEMATIZADOR NO FUNCIONA EN ESPAÑOL, HAY QUE USAR EL DE SPACY 
import spacy # PARA LEMATIZAR EN ESPAÑOL Y DESPUES PARA VECTORIZAR CON GLOVE
nlp_spacy = spacy.load("es_core_news_md")  # Carga el modelo en español # modelo pre entrenado para lematizacion y vectorizacion GLove
from nltk.tokenize import word_tokenize
from nltk import ngrams
#import pytesseract # para leer archivos en formato imagen
#from pdf2image import convert_from_path # para convertir los pdfs a imagen porque se leen mejor
import pdftotext
import numpy as np

## ELEGIR EL DATASET DE INFORMES

# Define the path of the zip file
#zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP1_2020q1\EA2_TP1_2020q1.zip".replace('\\', '/')
#zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP2_2020q1\EA2_TP2_2020q1.zip".replace('\\', '/')
#zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP3_2020q1\EA2_TP3_2020q1.zip".replace('\\', '/')
#zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP1_Plagios_simulados_2020q1\EA2_TP1_Plagios_simulados_2020q1.zip".replace('\\', '/')
#zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP2_Plagios_simulados_2020q1\EA2_TP2_Plagios_simulados_2020q1.zip".replace('\\', '/')
zip_file_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP3_Plagios_simulados_2020q1\EA2_TP3_Plagios_simulados_2020q1.zip".replace('\\', '/')


# Defino el path del enunciado
#enunciado_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP1_2020q1\EA2_TP1_2020q1_enunciado.pdf".replace('\\', '/')
#enunciado_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP2_2020q1\EA2_TP2_2020q1_enunciado.pdf".replace('\\', '/')
enunciado_path = r"C:\Users\HP\Desktop\Tesis\EA2_TP3_2020q1\EA2_TP3_2020q1_enunciado.pdf".replace('\\', '/')

###########

os.chdir("C:\\Users\HP\Desktop\Tesis")

# Define regular expression pattern to extract student name and ID from folder names
folder_pattern = re.compile(r'(.+)_([0-9]+)_.*')

# Create an empty DataFrame to store the results
#results_df = pd.DataFrame(columns=['Name', 'ID', 'Text', 'Owner'])

# Create an empty list to store the dictionaries
results_list = []

# Open the zip file
with zipfile.ZipFile(zip_file_path, mode = 'r') as zip_file:
    
    # Loop over all the files in the zip file
    for file_name in zip_file.namelist():
        
        # Check if the file is a PDF
        if file_name.endswith('.pdf'):
            
            # Extract the student name and ID from the folder name
            folder_name = os.path.dirname(file_name)
            match = folder_pattern.match(folder_name)
            if match:
                name, id = match.groups()
            else:
                print(f"Error: no se puede extraer Nombre y ID de la carpeta '{folder_name}'. Revisar formato de nombres de las carpetas.")
                continue
            
            # inicializo la lista de paginas del pdf
            paginas_pdf = []
            
            # Extract the text from the PDF
            with zip_file.open(file_name, mode = 'r') as pdf_file:
                pdf = pdftotext.PDF(pdf_file)
                for page in pdf:
                    paginas_pdf.append(page)

            # uno todas las páginas en un único string
            text = '\n'.join(paginas_pdf)
                
                
                # otra forma con pypdf2
                #pdf_data = pdf_file.read()
                #pdf_buffer = BytesIO(pdf_data) #es un binary buffer para leer en modo binario porque son PDFs (sino debería pasar a txt)
                #pdf_reader = PdfReader(pdf_file)
                #owner = pdf_reader.metadata.author
                #text = ''
                #for page_num in range(len(pdf_reader.pages)):
                #    page = pdf_reader.pages[page_num]
                #    text += page.extract_text()
            
            # Extract the owner from the PDF metadata
            #with zip_file.open(file_name) as pdf_file:
            #    pdf_reader = PdfReader(pdf_file)
            #    owner = pdf_reader.metadata().author
            
            # Add a new dictionary to the list
            results_list.append({'Name': name,
                                 'ID': id,
                                 'Text': text})#,
                                 #'Owner': owner})
            
            # Add a new row to the results DataFrame
            #results_df = results_df.append({'Name': name,
            #                                'ID': id,
            #                                'Text': text,
            #                                'Owner': owner}, ignore_index=True)
            
# Concatenate all the dictionaries into a single DataFrame
df = pd.concat([pd.DataFrame(d, index=[0]) for d in results_list], ignore_index=True)

# Print the results DataFrame
print(df.head())
df.iloc[0]["Text"]

# preprocesamiento primario de los textos
def preprocesamiento_primario_textos(text):
    
    # \n es el fin de una linea e texto
    # \r\n es el inicio de una linea de texto
    # \x0c es un salto de página
    # \n̂ es un salto de línea y un posible cambio de indentación en el código fuente
    
    # Remuevo los caracteres especiales y reemplazo por espacios " "
    text = text.replace(r'\n{1,}|\r{1,}|\x0c{1,}|\n̂{1,}', ' ')
    
    # Uso unicode que normaliza el texto, por ejemplo remueve tildes
    #text = unidecode(text) # ESTO VA A SER MEJOR HACERLO APARTE A LO ULTIMO PARA QUE NO AFECTE LA LEMATIZACIÓN
    
    # Convierto todas las palabras a minúscula
    text = text.lower()
    
    # Eliminar espacios en blanco al principio y al final
    text = text.strip()
    
    # Remuevo signos de puntuación
    #text = re.sub(r'', ' ', text)
    
    #Remuevo cualquier caracter que no sea letra, numero o espacio en blanco
    #text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remuevo espacios en blanco de más entre palabras
    text = re.sub(r'\s+', ' ', text)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    return text

df['Text'] = df['Text'].apply(preprocesamiento_primario_textos)

df.iloc[0]["Text"]


# Leo el enunciado para luego removerlo del texto del informe

# Create an empty list to store the text of each page
paginas_enunciado = []

# Open the PDF file and read in the text of each page
with open(enunciado_path, mode='rb') as f:
    pdf = pdftotext.PDF(f)
    for page in pdf:
        paginas_enunciado.append(page)

# uno todas las páginas en un único string
enunciado = '\n'.join(paginas_enunciado)

# enunciado
enunciado

# con pypdf2
#pdf_reader_enunciado = PdfReader(enunciado_path)
#enunciado = ''
#for page_num in range(len(pdf_reader_enunciado.pages)):
#    page = pdf_reader_enunciado.pages[page_num]
#    enunciado += page.extract_text()

# con tessearct, leo el enunciado en pdf pero antes conviertiendolo a imagen
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#images = convert_from_path(enunciado_path)
#ocr_text = ''
#for i in range(len(images)):        
#    page_content = pytesseract.image_to_string(images[i])
#    page_content = '***PDF Page {}***\n'.format(i+1) + page_content
#    ocr_text = ocr_text + ' ' + page_content

def preprocesamiento_enunciado(enunciado):
    
    # Los inicio de parrafos los cambio por parrafo_nuevo para despues tokenizar
    enunciado = enunciado.replace('\r\n', ' parrafo_nuevo ')
    
    # Remuevo los caracteres especiales y reemplazo por espacios en blanco
    enunciado = enunciado.replace('\x0c', ' ')
    enunciado = enunciado.replace('\n', ' ')
    enunciado = enunciado.replace('\n̂', ' ')
    
    # Convierto todas las palabras a minúscula
    enunciado = enunciado.lower()
    
    # Eliminar espacios en blanco al principio y al final
    enunciado = enunciado.strip()
    
    # Remuevo espacios en blanco de más entre palabras
    enunciado = re.sub(r'\s+', ' ', enunciado)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    enunciado = re.sub(r'\s+([.,;:])', r'\1', enunciado)
    
    # normalizo el texto
    #enunciado = unidecode(enunciado) # ESTO NO HARIA FALTA PARA EL ENUNCIADO PORQUE TODAVIA NO LO HAGO PARA LOS TEXTOS
    
    return enunciado

enunciado = preprocesamiento_enunciado(enunciado)

enunciado

# Tokenizo el enunciado en oraciones y parrafos
tokenizador_oraciones_parrafos = re.compile(r'parrafo_nuevo|\n{1,}|[.:!?•]\s+|[a-zA-Z]\)')
enunciado_oraciones_parrafos = tokenizador_oraciones_parrafos.split(enunciado)

enunciado_oraciones_parrafos

# elimno aquellos tokens que son items, es decir una letra o un número
def eliminar_tokens_una_letra_numero_items(tokens):
    """
    Elimina los tokens que sean una única letra o número o nada
    """
    return [token for token in tokens if len(token) > 1 or not token.isalnum()]

enunciado_oraciones_parrafos = eliminar_tokens_una_letra_numero_items(enunciado_oraciones_parrafos)

# elimino aquellos tokens que son una única palabra porque podría generar eliminar palabras no debidas en el informe
def eliminar_tokens_una_palaba(tokens):
    """
    Elimina los tokens que sean una única palabra
    """
    tokens_filtrados = []
    
    for token in tokens:
        if len(token.split()) > 1:
            tokens_filtrados.append(token)
    
    return tokens_filtrados

enunciado_oraciones_parrafos = eliminar_tokens_una_palaba(enunciado_oraciones_parrafos)

# Eliminar espacios en blanco al principio y al final
def eliminar_tokens_espacios(tokens):
    """
    Elimina los espacios en blanco al inicio y fin
    """
    tokens_sin_espacios = []
    
    for token in tokens:
        token = token.strip()
        tokens_sin_espacios.append(token)
    
    return tokens_sin_espacios

enunciado_oraciones_parrafos = eliminar_tokens_espacios(enunciado_oraciones_parrafos)

print(enunciado_oraciones_parrafos)

len(df.iloc[0]["Text"]) # largo de un texto antes de sacar el enunciado

# Ahora elimino estos tokens almacenados en enunciado_oraciones_parrafos de los textos de los informes
for i, row in df.iterrows():
    for token in enunciado_oraciones_parrafos:
        df.at[i, 'Text'] = df.at[i, 'Text'].replace(token, '')

#for token in enunciado_oraciones_parrafos:
#    #df['Text'] = df['Text'].str.replace(token, '')
#    df['Text'] = df['Text'].replace(token, '')

len(df.iloc[0]["Text"]) # largo de un texto despues de sacar el enunciado

# hago pruebas para buscar cadenas de texto del enunciado en los tps, para verificar si se elimina el enunciado
#patron = r'5\.000 \$\/ha'
#patron2 = r'5.000 $ha'
#patron4 = r'a partir de la información obtenida, construya el intervalo de confianza del 95% para la cantidad total de madera en el bosque'
#patron5 = r'el tamaño de muestra global'

#for j in range(108):
#    
#    resultado = re.search(patron5, df.iloc[j]["Text"])
#    print(j)
#    if resultado:
#        print("Se encontró la cadena de texto:", resultado.group())
#    else:
#        print("No se encontró la cadena de texto.")


# textos con un preprocesamiento inicial
df.iloc[0]['Text']

df_pri = df.copy()

# data frames de textos con distintos preprocesamientos
# df_pri únicamente con preproc PRIMARIO
# df_sec únicamente con preproc PRIMARIO y SECUNDARIO
# df_ter únicamente con preproc PRIMARIO, SECUNDARIO y TERCIARIO--> para tokenización con n gramas, palabras, ORACIONES

# pre procesamiento SECUNDARIO de los textos: 
# eliminar completamente todo número o signo de formula matematica
# elimino todo lo que no sea letra, espacio o punto (porque despues tokenizo en oraciones, pero elimino el resto de signos de puntuación)
def preprocesamiento_secundario_textos(text):

    # Elimino cualquier caracter que no sea letra, espacio o punto # esto eliminaria numeros y signos de operaciones matemáticas
    text = re.sub('[^A-Za-z\sñáéíóúüÁÉÍÓÚÜ\.]+', '', text)
    
    # Eliminar espacios en blanco al principio y al final
    text = text.strip()

    # Remuevo espacios en blanco de más entre palabras
    text = re.sub(r'\s+', ' ', text)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    # Reemplazo varios puntos consecutivos por uno solo (estaría considerando que al llegar a una fórmula, la oración termina)
    text = re.sub('\.+', '.', text)
    
    return text

df_sec = df_pri.copy()

df_sec['Text'] = df_sec['Text'].apply(preprocesamiento_secundario_textos)

df_sec.iloc[0]["Text"]

# pre procesamiento TERCIARIO de los textos:
# LEMATIZACIÓN y REMOCIÓN DE STOPWORDS

df_ter = df_sec.copy()

# Descargar los recursos necesarios de NLTK # con descargarlos una vez seria suficiente
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Definir las stopwords en español 
# como MEJORA se podría agregar a esta lista palabras típicas en un informe de la materia
spanish_stopwords = stopwords.words('spanish')

# Inicializar el lematizador de NLTK # USO EL DE SPACY
#lemmatizer = WordNetLemmatizer()

# cargo el modelo de spacy
#nlp_spacy = spacy.load("es_core_news_lg")  # Carga el modelo en español # LO CARGO AL PRINCIPIO

# Definir una función para lematizar y remover stopwords
def lemmatizar_y_remover_stopwords(text): # ACA LUEGO DE ESTO PASO A UNIDECODE PORQUE YA ESTARIA PREPROCESADO
    
    # Procesa el texto con spaCy
    doc = nlp_spacy(text)
    
    # Remueve las stopwords y lematiza
    tokens_procesados = [token.lemma_ for token in doc if not token.is_stop]
    
    # vuelvo a unir tokens con un espacio
    clean_text = " ".join(tokens_procesados)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    clean_text = re.sub(r'\s+([.,;:])', r'\1', clean_text)
    
    # normalizo el texto con unidecode
    clean_text = unidecode(clean_text)
    
    # elimino las palabras "version" y "estudiantil" porque es lo que figura en las capturas de pantalla de InfoStat
    clean_text = clean_text.replace("version", "")
    clean_text = clean_text.replace("estudiantil", "")
    
    # elimino palabras conformadas únicamente por una letra que no esté en el listado de palabras de una letra (esto es por si quedaron caracteres basura sueltos, puede haber por las formulas matematicas)
    palabras = clean_text.split()
    palabras_filtradas = [palabra for palabra in palabras if (len(palabra) > 1 or palabra.lower() in ['a', 'e', 'y', 'o', 'u'])]
    # uno las palabras filtradas de nuevo en un solo texto
    clean_text = ' '.join(palabras_filtradas)
    
    return clean_text

# Aplicar la función a la columna 'Text' del DataFrame
df_ter['Text'] = df_ter['Text'].apply(lemmatizar_y_remover_stopwords)


# antes del tercer preprocesamiento
df_sec.iloc[0]['Text']

# despues del tercer preprocesamiento
df_ter.iloc[0]['Text']


'''
IMPORTANTE!!!

TODAVIA NO SACO LOS PUNTOS . EN EL TEXTO PROCESADO PORQUE ANTES TENGO QUE HACER LA TOKENIZACIÓN DE ORACIONES, DESPUES YA LO PUEDO HACER
'''

#df_ter_bis = df_sec.copy()



##################################################### TOKENIZACIÓN de los textos

# Inicializar el tokenizador de oraciones
tokenizador_oraciones = re.compile(r'\.')
def tokenizador_textos_oraciones(text):
    
    # tokenizo en oraciones ante la presencia de un punto
    tokens_oraciones = tokenizador_oraciones.split(text)
    
    # elimino tokens que sean una letra o número
    tokens_oraciones = eliminar_tokens_una_letra_numero_items(tokens_oraciones)
    
    # elimino tokens que sean una sola palabra
    tokens_oraciones = eliminar_tokens_una_palaba(tokens_oraciones)
    
    # elimno tokens que sean espacios vacios
    tokens_oraciones = eliminar_tokens_espacios(tokens_oraciones)
    
    return tokens_oraciones

df_ter['Tokens_oraciones'] = df_ter['Text'].apply(tokenizador_textos_oraciones)

# obtengo los tokens de oraciones
df_ter.iloc[0]['Tokens_oraciones']
df_ter.iloc[1]['Tokens_oraciones']
df_ter.iloc[2]['Tokens_oraciones']
df_ter.iloc[3]['Tokens_oraciones']

''' ESTO YA NO TIENE SENTIDO PORQUE LO HAGO ANTES
# Función para lematizar y remover stopwords de una lista de tokens
def lemmatizar_y_remover_stopwords_tokens_oraciones(tokens):
    
    # lematizo y remuevo stopwords de cada token (oración)
    tokens = [lemmatizar_y_remover_stopwords(token) for token in tokens]
    
    # tengo que volver a eliminar tokens que sean una sola palabra
    tokens = eliminar_tokens_una_palaba(tokens)
    
    return tokens

# Aplicar la función process_text a cada fila del dataframe df
df_ter_bis['Tokens_oraciones'] = df_ter_bis['Tokens_oraciones'].apply(lemmatizar_y_remover_stopwords_tokens_oraciones)

# obtengo los tokens de oraciones lematizados y con stopwords removidas
df_ter_bis.iloc[0]['Tokens_oraciones']
df_ter_bis.iloc[1]['Tokens_oraciones']
df_ter_bis.iloc[2]['Tokens_oraciones']
df_ter_bis.iloc[3]['Tokens_oraciones']
'''

# ahora que ya realice la tokenización de oraciones, puedo sacarle los puntos al texto
def ultimo_preprocesamiento_elimino_puntos(text):
    
    # elimino los puntos
    clean_text = re.sub(r'\.', '', text)
    
    # DE VUELTA (PORQUE AL SACAR PUNTOS VUELVE A HABER) elimino palabras conformadas únicamente por una letra que no esté en el listado de palabras de una letra (esto es por si quedaron caracteres basura sueltos, puede haber por las formulas matematicas)
    palabras = clean_text.split()
    palabras_filtradas = [palabra for palabra in palabras if (len(palabra) > 1 or palabra.lower() in ['a', 'e', 'y', 'o', 'u'])]
    # uno las palabras filtradas de nuevo en un solo texto
    clean_text = ' '.join(palabras_filtradas)
    
    return clean_text

df_ter['Text'] = df_ter['Text'].apply(ultimo_preprocesamiento_elimino_puntos)

# obtengo los tokens de oraciones
df_ter.iloc[0]['Text']
df_ter.iloc[1]['Text']
df_ter.iloc[2]['Text']
df_ter.iloc[3]['Text']


# ARMO UN UNICO DF CON LAS TOKENIZACIONES

# tokenizo oraciones para Jaccard Oraciones y Jaccard Ponderado Oraciones
# tokenizo n=5 gramas solapados para Jaccard
# tokenizo n=7 gramas solapados para Jaccard
# tokenizo palabras para Jaccard y técnicas de Bag of Words
# en algunos casos voy a necesitar el texto sin tokenizar
# buscar distancias como el coseno, vectorizar, etc

# armo un df único con cada tokenización
df_tokens = pd.DataFrame({
    'Nombre': df_ter['Name'],
    'ID': df_ter['ID'],
    'Texto': df_ter['Text'],
    'Tokens_oraciones': df_ter['Tokens_oraciones']
})

# incluyo tokenización de palabras
def tokenizador_palabras(text):
    # remuevo los puntos
    text = text.replace(".", "")
    # tokenizo
    tokens = word_tokenize(text.lower(), language='spanish')
    return tokens

df_tokens['Tokens_palabras'] = df_tokens['Texto'].apply(tokenizador_palabras)

# incluyo tokenizaciones de n gramas con n=5, n=7 y n=9
def tokenizador_ngramas(tokens_palabras, n):
    tokens_ngrams = list(ngrams(tokens_palabras, n))
    tokens_ngrams = [' '.join(list(tupla)) for tupla in tokens_ngrams]
    return tokens_ngrams

df_tokens['Tokens_3gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 3))
df_tokens['Tokens_5gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 5))
df_tokens['Tokens_7gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 7))
df_tokens['Tokens_9gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 9))

df_tokens.iloc[0]['Tokens_3gramas']
df_tokens.iloc[0]['Tokens_5gramas']
df_tokens.iloc[0]['Tokens_7gramas']
df_tokens.iloc[0]['Tokens_9gramas']


################################################### Armo una tokenizacion aparte para el entrenamiento de redes neuronales con word2vec
'''
# oraciones tokenizadas en lista para word2vec
oraciones__word2vec = list(df_tokens['Tokens_oraciones'])
oraciones_tokenizadas_word2vec = []
for texto in oraciones__word2vec:
    for oracion in texto:
        oracion_tokenizada = nltk.word_tokenize(oracion)
        oraciones_tokenizadas_word2vec.append(oracion_tokenizada)

# textos en lista para doc2vec
palabras_doc2vec = list(df_tokens['Tokens_palabras'])

# exporto los datos en formato json
import json
'''
# Guardar las listas en un json en las carpetas correspondientes
'''
################## EA2_TP1_2020q1
# para word2vec
with open("data_preprocesada_entrenamiento_word2vec\oraciones_tokenizadas_EA2_TP1_2020q1.json", "w") as archivo:
    json.dump(oraciones_tokenizadas_word2vec, archivo)

# para doc2vec
with open("data_preprocesada_entrenamiento_doc2vec\\palabras_EA2_TP1_2020q1.json", "w") as archivo:
    json.dump(palabras_doc2vec, archivo)
'''
'''
################## EA2_TP2_2020q1
# para word2vec
with open("data_preprocesada_entrenamiento_word2vec\oraciones_tokenizadas_EA2_TP2_2020q1.json", "w") as archivo:
    json.dump(oraciones_tokenizadas_word2vec, archivo)

# para doc2vec
with open("data_preprocesada_entrenamiento_doc2vec\\palabras_EA2_TP2_2020q1.json", "w") as archivo:
    json.dump(palabras_doc2vec, archivo)
'''
'''
################## EA2_TP3_2020q1
# para word2vec
with open("data_preprocesada_entrenamiento_word2vec\oraciones_tokenizadas_EA2_TP3_2020q1.json", "w") as archivo:
    json.dump(oraciones_tokenizadas_word2vec, archivo)

# para doc2vec
with open("data_preprocesada_entrenamiento_doc2vec\\palabras_EA2_TP3_2020q1.json", "w") as archivo:
    json.dump(palabras_doc2vec, archivo)
'''

########################################################## VECTORIZACIONES

# Vectorizaciones
# CountVectorizer
# TfidfVectorizer
# Doc2Vec o Word2vec (CBOW y Skip-gram)

# vectorizo los textos y/o los tokens ya que es necesario para el calculo de algunas distancias
df_tokens.columns


################ Vectorización CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# paso los textos preprocesados a una lista
lista_textos = list(df_tokens['Texto'])

#### para palabras (1gramas), 3gramas, 5gramas, 7gramas, 9gramas # CONSIDERAR VECTORIZAR CON NGRAMAS COMBINADOS (ej ngram_range=(1, 3))
for k in [1, 3, 5, 7, 9]:
    # inicializo el vectorizador
    count_vectorizer = CountVectorizer(ngram_range=(k, k))

    # ajuste de la vectorización en base a los textos
    textos_count_vectorizer_transformados = count_vectorizer.fit_transform(lista_textos)
    
    # veo los nombreS de los tokens unicos identificados
    count_vectorizer.vocabulary_
    
    # agrego la columna con el texto vectorizado # LOS AGREGO COMO LISTAS PERO DESPUES PARA USARLOS HAY QUE PASARLOS DE VUELTA A ARRAY
    if k==1:
        df_tokens['vectorizado_CountVectorizer_palabras'] = list(textos_count_vectorizer_transformados.toarray())
    else:
        df_tokens['vectorizado_CountVectorizer_'+str(k)+'gramas'] = list(textos_count_vectorizer_transformados.toarray())


################ Vectorización TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# paso los textos preprocesados a una lista
lista_textos = list(df_tokens['Texto'])

#### para palabras (1gramas), 3gramas, 5gramas, 7gramas, 9gramas
for k in [1, 3, 5, 7, 9]:
    # inicializo el vectorizador
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(k, k))

    # ajuste de la vectorización en base a los textos
    textos_tfidf_vectorizer_transformados = tfidf_vectorizer.fit_transform(lista_textos)
    
    # veo los nombreS de los tokens unicos identificados
    tfidf_vectorizer.vocabulary_
    
    # agrego la columna con el texto vectorizado # LOS AGREGO COMO LISTAS PERO DESPUES PARA USARLOS HAY QUE PASARLOS DE VUELTA A ARRAY
    if k==1:
        df_tokens['vectorizado_TfidfVectorizer_palabras'] = list(textos_tfidf_vectorizer_transformados.toarray())
    else:
        df_tokens['vectorizado_TfidfVectorizer_'+str(k)+'gramas'] = list(textos_tfidf_vectorizer_transformados.toarray())


# veo las columnas agregadas
df_tokens.columns

# prueba de distancia euclidesa entre vectores
#np.sqrt((df_tokens['vectorizado_TfidfVectorizer_palabras'][0] - df_tokens['vectorizado_TfidfVectorizer_palabras'][1])^{2})



################ Vectorización Doc2vec

# CBOW (Continuos Bag Of Words)
# Skip-gram

#import gensim
#from gensim.models import Word2Vec
from gensim.models import Doc2Vec

# cargo un modelo
modelo_doc2vec = Doc2Vec.load("C:\\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\modelo_doc2vec_informes_vectors50_window2_minc2_dm0_epochs40.model")

# prueba de vectorizar
#modelo_doc2vec.infer_vector(df_tokens['Tokens_palabras'][0])

# defino funcion para generar vectores Doc2Vec
def vectorizar_texto_Doc2Vec(tokens_palabras):
    
    vector = modelo_doc2vec.infer_vector(tokens_palabras)
    
    vector = vector.reshape(1, -1) # porque despues calculo la similitud de coseno
    
    return vector

# vectorizo Doc2Vec
df_tokens['vectorizado_Doc2Vec'] = df_tokens['Tokens_palabras'].apply(vectorizar_texto_Doc2Vec)

# vectores
df_tokens['vectorizado_Doc2Vec']

########################################################## DISTANCIAS/SIMILITUDES ENTRE TEXTOS

# Distancias/Similitudes
# Jaccard = (A & B) / (A o B)
# Coseno = (A . B) / (||A||*||B||)
# Euclidea = raiz(A^{2}+B^{2}) # puede que ni tenga sentido calcularla por la distorsion que genera la diferencia de long en los textos
# Damerau-Levenshtein = 1- (MIN(cantidad de operaciones necesarias para que los conjuntos A y B sean iguales)/MAX(long_A; long_B))
# Jaro-Winkler # no se si esta vale la pena porque tiene un poco de la de jaccad y de la de Damerau-Levenshtein
# Hamming (solo mencionar como caso particular de Levenshtein)
# Longest Common Subsequence (la subsecuencia que busca no es corrida)
# Longest Common Substring (la subsecuencia si es corrida)
# (las cuatro ultimas son parte de las Edit Distance, todas tienen el principio de cuanto hay que modificar una cadena para que sea igual a otra)

# Defino las funciones de calculo de distancia entre los textos o tokens
df_tokens.columns

#from sklearn.metrics import jaccard_score
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from fastDamerauLevenshtein import damerauLevenshtein
#import statistics


################################ Jaccard
# cantidad de coincidencias (intersección) entre tokens/union total de tokens 
def jaccard_similarity(tokens1, tokens2):
    set1=set(tokens1)
    set2=set(tokens2)
    largo_interseccion = len(set1.intersection(set2))
    largo_union = len(set1.union(set2))
    return largo_interseccion / largo_union


# Calculo los pares de similitudes de Jaccard, tokens palabras
columna_df = 'Tokens_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_palabras_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_palabras'])


# Calculo los pares de similitudes de Jaccard, tokens 3 gramas
columna_df = 'Tokens_3gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_3gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_3gramas'])



# Calculo los pares de similitudes de Jaccard, tokens 5 gramas
columna_df = 'Tokens_5gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_5gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_5gramas'])



# Calculo los pares de similitudes de Jaccard, tokens 7 gramas
columna_df = 'Tokens_7gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_7gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_7gramas'])



# Calculo los pares de similitudes de Jaccard, tokens 9 gramas
columna_df = 'Tokens_9gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_9gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_9gramas'])


# Calculo los pares de similitudes de Jaccard, tokens oraciones
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    tokens1 = set(df_tokens[columna_df][i])
    tokens2 = set(df_tokens[columna_df][j])
    jaccard_sim = jaccard_similarity(tokens1, tokens2)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], jaccard_sim])

# Crear un DataFrame con la matriz de similitud
pares_jaccard_oraciones_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_oraciones'])



# Jaccard Ponderado (para Oraciones o ngramas largos): (lo inventé yo)
# que le de más peso a las coincidencias de oraciones más largas que a las más cortas
# SUMA(largo de oraciones coincidentes)/SUMA(total de largo de oraciones en el texto)



################################ Coseno para COUNT VECTORIZER

df_tokens.columns

# Calculo los pares de similitudes de Coseno, tokens palabras
columna_df = 'vectorizado_CountVectorizer_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens['vectorizado_CountVectorizer_palabras'][i]], [df_tokens['vectorizado_CountVectorizer_palabras'][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_palabras_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_CountVec_palabras'])



# Calculo los pares de similitudes de Coseno, tokens 3gramas
columna_df = 'vectorizado_CountVectorizer_3gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens['vectorizado_CountVectorizer_3gramas'][i]], [df_tokens['vectorizado_CountVectorizer_3gramas'][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_3gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_CountVec_3gramas'])



# Calculo los pares de similitudes de Coseno, tokens 5gramas
columna_df = 'vectorizado_CountVectorizer_5gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens['vectorizado_CountVectorizer_5gramas'][i]], [df_tokens['vectorizado_CountVectorizer_5gramas'][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_5gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_CountVec_5gramas'])



# Calculo los pares de similitudes de Coseno, tokens 7gramas
columna_df = 'vectorizado_CountVectorizer_7gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens['vectorizado_CountVectorizer_7gramas'][i]], [df_tokens['vectorizado_CountVectorizer_7gramas'][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_7gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_CountVec_7gramas'])



# Calculo los pares de similitudes de Coseno, tokens 9gramas
columna_df = 'vectorizado_CountVectorizer_9gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens['vectorizado_CountVectorizer_9gramas'][i]], [df_tokens['vectorizado_CountVectorizer_9gramas'][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_9gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_CountVec_9gramas'])


################################ Coseno para TF-IDF VECTORIZER

df_tokens.columns

# Calculo los pares de similitudes de Coseno, tokens palabras
columna_df = 'vectorizado_TfidfVectorizer_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens[columna_df][i]], [df_tokens[columna_df][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_palabras_TFIDF_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_TFIDFVec_palabras'])



# Calculo los pares de similitudes de Coseno, tokens 3gramas
columna_df = 'vectorizado_TfidfVectorizer_3gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens[columna_df][i]], [df_tokens[columna_df][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_3gramas_TFIDF_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_TFIDFVec_3gramas'])



# Calculo los pares de similitudes de Coseno, tokens 5gramas
columna_df = 'vectorizado_TfidfVectorizer_5gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens[columna_df][i]], [df_tokens[columna_df][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_5gramas_TFIDF_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_TFIDFVec_5gramas'])



# Calculo los pares de similitudes de Coseno, tokens 7gramas
columna_df = 'vectorizado_TfidfVectorizer_7gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens[columna_df][i]], [df_tokens[columna_df][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_7gramas_TFIDF_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_TFIDFVec_7gramas'])



# Calculo los pares de similitudes de Coseno, tokens 9gramas
columna_df = 'vectorizado_TfidfVectorizer_9gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity([df_tokens[columna_df][i]], [df_tokens[columna_df][j]])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])

# Crear un DataFrame con la matriz de similitud
pares_coseno_9gramas_TFIDF_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_TFIDFVec_9gramas'])



################################ Coseno para Doc2Vec

# Calculo los pares de similitudes de Coseno, vectorizacion Doc2Vec
columna_df = 'vectorizado_Doc2Vec'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = cosine_similarity(df_tokens[columna_df][i], df_tokens[columna_df][j])[0][0]
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])


# Crear un DataFrame con la matriz de similitud
pares_coseno_Doc2Vec_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_Doc2Vec'])



#----------------------------------------------- hago un merge de las metricas calculadas hasta el momento y guardo el excel

#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
#nombre_base_datos = 'EA2_TP3_2020q1'
#nombre_base_datos = 'EA2_TP1_Plagios_simulados_2020q1'
#nombre_base_datos = 'EA2_TP2_Plagios_simulados_2020q1'
nombre_base_datos = 'EA2_TP3_Plagios_simulados_2020q1'


dfs_similitudes_lista = [pares_jaccard_palabras_df, pares_jaccard_3gramas_df, pares_jaccard_5gramas_df, pares_jaccard_7gramas_df, pares_jaccard_9gramas_df, pares_jaccard_oraciones_df,
                         pares_coseno_palabras_df, pares_coseno_3gramas_df, pares_coseno_5gramas_df, pares_coseno_7gramas_df, pares_coseno_9gramas_df, 
                         pares_coseno_palabras_TFIDF_df, pares_coseno_3gramas_TFIDF_df, pares_coseno_5gramas_TFIDF_df, pares_coseno_7gramas_TFIDF_df, pares_coseno_9gramas_TFIDF_df,
                         pares_coseno_Doc2Vec_df]

merged_df = dfs_similitudes_lista[0] # primer df
merged_df['ID_unico'] = nombre_base_datos + '_' + merged_df['ID1'].astype(str) + '_' + merged_df['ID2'].astype(str)
merged_df = merged_df.drop(['ID1', 'ID2'], axis=1)
# Itero a través de los DataFrames y agrego la nueva columna "ID_único"
for df in dfs_similitudes_lista[1:]:
    # agrego un ID unico
    df['ID_unico'] = nombre_base_datos + '_' + df['ID1'].astype(str) + '_' + df['ID2'].astype(str)
    # hago un merge
    merged_df = pd.merge(merged_df, df.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')
id_unico = merged_df.pop('ID_unico')
merged_df.insert(0, 'ID_unico', id_unico)

# exporto las métricas calculadas hasta ahora
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_Plagios_simulados_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_Plagios_simulados_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_Plagios_simulados_2020q1.xlsx", index=False)

# importo las metricas para no tener que volver a calcularlas
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx")
merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx")

################################ Coseno para Vectorizaciones Word2Vec PREENTRENADOS

# directamente puedo usar spacy en los textos preprocesados, por detras se usa el modelo de word embedding preentrenado de glove

# CON EL MODELO PRE ENTRENADO LARGE TARDA BASTANTE, PROBAR CON EL MEDIUM # EFECTIVAMENTE EL MEDIUM ES MAS RAPIDO PERO TARDA

'''
No parece que funcione bien el calculo de un texto versus el otro

podria pensar en calcular similitud entre cada oracion


Probablemente lo que sucede es que al estar representando en el espacio todas estas palabras
se genera un ruido por el hecho de que dos palabras de un mismo tema o genero van a estar cercanas en el espacio
pero por mas de que sean cercanas en el espacio por ser de la misma familia, no las hace sinónimos y mucho menos iguales.
Esto hace que al hacer un promedio del vector de cada palabra del texto, se pierda el peso de las palabras y oraciones
que son exactamente iguales, frente al peso de las palabras que son cercanas por ser del mismo tema, género o familia.
Es por eso que al tomar un enfoque como este puedo estar alejandome del grado de ofuscación que quiero
y me estoy acercando indeseablemente a un grado de ofuscación más cercano a la detección de tópicos e ideas similares.

Tambien es muy probable que el hecho de usar un modelo pre entrenado para esto no sea lo mas conveniente, ya que los datos
de entrenamiento de los word embeddings pueden no ser verdaderamente representativos de los textos a vectorizar.

'''

def similitud_coseno_Word2Vec_preentrenado(texto1, texto2):
    
    doc1 = nlp_spacy(texto1)
    doc2 = nlp_spacy(texto2)
    
    return doc1.similarity(doc2)


columna_df = 'Texto'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    coseno_sim = similitud_coseno_Word2Vec_preentrenado(df_tokens[columna_df][i], df_tokens[columna_df][j])
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], coseno_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_coseno_Word2Vec_preentrenado_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Coseno_Word2Vec_preentrenado'])

# agrego la columna ID_unico, hago merge con el resto de metricas y guardo en excel
#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
nombre_base_datos = 'EA2_TP3_2020q1'
pares_coseno_Word2Vec_preentrenado_df['ID_unico'] = nombre_base_datos + '_' + pares_coseno_Word2Vec_preentrenado_df['ID1'].astype(str) + '_' + pares_coseno_Word2Vec_preentrenado_df['ID2'].astype(str)
merged_df = pd.merge(merged_df, pares_coseno_Word2Vec_preentrenado_df.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')

#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)

# importo las metricas para no tener que volver a calcularlas
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx")
merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx")

'''
################################ Diferencia de longitud de los textos (por largo de cadenas de textos, por cantidad de palabras, y por cantidad de oraciones)

ESTO DE LAS DIFERENCIAS DE LONGITUD NO LO HACEMOS PORQUE NO TIENE SENTIDO, POR MAS DE QUE DE ACA SIGNIFICATIVO PUEDE QUE EN GENERAL NO LO SEA, NO TIENE FUNDAMENTO TEORICO

# El hecho de que el coseno no tenga en cuenta el largo de los textos como sí sucede con la euclidea, parece una ventaja a simple vista
# pero en realidad no tenemos una idea verdadera de si la direferencia en el largo de dos textos es una variable significativa en la deteccion de plagios
# es decir, la hipotesis a comprobar o rechazar es si a mayor o menor diferencia entre el largo de los textos existe una mayor probabilidad de copia
# es por eso que calculamos esta diferencia entre el largo de los pares de textos y la usaremos como una variable explicativa más

# calculo los pares de diferencias de largo entre las cadenas de textos
columna_df = 'Texto'
num_textos = len(df_tokens)
dif_largo_list = []
for i, j in combinations(range(num_textos), 2):
    dif_largo_list.append([df_tokens['ID'][i], df_tokens['ID'][j], abs(len(df_tokens[columna_df][i]) - len(df_tokens[columna_df][j]))])

# Crear un DataFrame con la matriz de similitud
pares_diferencia_largo_textos_df = pd.DataFrame(dif_largo_list, columns=['ID1', 'ID2', 'Diferencia_largo_texto'])


# calculo los pares de diferencias de largo entre las listas de tokens de palabras (diferencia cant de palabras)
columna_df = 'Tokens_palabras'
num_textos = len(df_tokens)
dif_largo_list = []
for i, j in combinations(range(num_textos), 2):
    dif_largo_list.append([df_tokens['ID'][i], df_tokens['ID'][j], abs(len(df_tokens[columna_df][i]) - len(df_tokens[columna_df][j]))])

# Crear un DataFrame con la matriz de similitud
pares_diferencia_cant_palabras_df = pd.DataFrame(dif_largo_list, columns=['ID1', 'ID2', 'Diferencia_cant_palabras'])


# calculo los pares de diferencias de largo entre las listas de tokens de oraciones (diferencia cant de oraciones)
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
dif_largo_list = []
for i, j in combinations(range(num_textos), 2):
    dif_largo_list.append([df_tokens['ID'][i], df_tokens['ID'][j], abs(len(df_tokens[columna_df][i]) - len(df_tokens[columna_df][j]))])

# Crear un DataFrame con la matriz de similitud
pares_diferencia_cant_oraciones_df = pd.DataFrame(dif_largo_list, columns=['ID1', 'ID2', 'Diferencia_cant_oraciones'])

'''

################################ Damerau-Levenshtein

df_tokens.columns

# Dos formas de calcular similitudes entre los textos con esta distancia:
# 1) "Indirecto": calcular la similitud damerau-leven entre cada oracion de un texto versus cada oracion de un segundo texto
# 2) "Directo": calcular la similitud damerau-leven de un texto completo versus otro texto completo (esto puede ser bastante exigente computacionalmente con tokenizaciones de ngramas de alto n)

# en estas dos metodologias se puede utilizar distintas tokenizaciones, así como distintas ponderaciones en el costo de las operaciones

# ambas son muy exigentes computacionalmente
# el metodo 1) sobretodo tarda mucho, IDEA PARA MEJORAR: calcular la simil de jaccard previamente y solo calcular dam-lev para valores de jaccard mayores a 0
# NO FUNCIONÓ LO ANTERIOR, AUN ASI TARDA DEMASIADO

'''
# 1)
# funcion para damerau-levenshtein entre oraciones (luego se calcularia una media o promedio, o incluso cuantos valores son mayores a un cierto corte)
def damerau_levenshtein_similarity_ORACIONES(oracion_1, oracion_2, similitud, tokenizacion, costo_extraer, costo_insertar, costo_reemplazar, costo_intercambiar):
    # las oraciones ya estan preprocesadas (lo mas importante es la lematización y remoción de stopwords)
    
    if tokenizacion =='palabras':
        # tokenizo las oraciones en palabras
        tokens_1 = word_tokenize(oracion_1, language='spanish')
        tokens_2 = word_tokenize(oracion_2, language='spanish')
        
    if tokenizacion =='2gramas':
        # tokenizo las oraciones en 2gramas
        tokens_palabras_1 = word_tokenize(oracion_1, language='spanish')
        tokens_palabras_2 = word_tokenize(oracion_2, language='spanish')
        tokens_1 = tokenizador_ngramas(tokens_palabras_1, 2)
        tokens_2 = tokenizador_ngramas(tokens_palabras_2, 2)


    # como el calculo de hacer todas las oraciones versus todas las oraciones del otro texto es muy exigente computacionalmente
    # utilizo la simil de jaccard para "descartar" los casos que ya sabemos que no tienen ninguna coincidencia
    #simil_jaccard = jaccard_similarity(tokens_1, tokens_2)    

    #if simil_jaccard <= 0.00001: # NO FUNCIONÓ, AUN ASI TARDA DEMASIADO
    #    dam_lev_sim = 0
    
        
    dam_lev_sim = damerauLevenshtein(tokens_1, tokens_2,
                                     similarity=similitud,
                                     deleteWeight = costo_extraer,
                                     insertWeight = costo_insertar,
                                     replaceWeight = costo_reemplazar, 
                                     swapWeight = costo_intercambiar)
    
    return dam_lev_sim

# ejemplo de uso entre ORACIONES
damerau_levenshtein_similarity_ORACIONES(str(df_tokens['Tokens_oraciones'][0][0]), str(df_tokens['Tokens_oraciones'][1][0]),
                                         similitud = True,
                                         tokenizacion = 'palabras',
                                         costo_extraer=1,
                                         costo_insertar=1,
                                         costo_reemplazar=1,
                                         costo_intercambiar=1)


# Calculo los pares de similitudes de Dam-Leven indirecto entre oraciones, tokens palabras, ponderaciones iguales
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    # hago las combinaciones entre cada oracion de un texto con cada oracion del otro
    similitudes_entre_oraciones = []
    for oracion1 in df_tokens[columna_df][i]:
        for oracion2 in df_tokens[columna_df][j]:
            dam_leven_dir_sim_ORACIONES = damerau_levenshtein_similarity_ORACIONES(oracion1, oracion2,
                                                                         tokenizacion='palabras',
                                                                         similitud = True,
                                                                         costo_extraer=1,
                                                                         costo_insertar=1,
                                                                         costo_reemplazar=1,
                                                                         costo_intercambiar=1)
            similitudes_entre_oraciones.append(dam_leven_dir_sim_ORACIONES)
    
    # promedio
    dam_leven_dir_sim_PROMEDIO = statistics.mean(similitudes_entre_oraciones)
    
    # mediana aritmetica
    dam_leven_dir_sim_MEDIANA = statistics.median(similitudes_entre_oraciones)
    
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim_PROMEDIO, dam_leven_dir_sim_MEDIANA])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_palabras_indirecto_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_INDIRECTO_palabras_PROMEDIO', 'Similitud_Damerau_Levenshtein_INDIRECTO_palabras_MEDIANA'])



# Calculo los pares de similitudes de Dam-Leven indirecto entre oraciones, tokens 2gramas, ponderaciones iguales
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    # hago las combinaciones entre cada oracion de un texto con cada oracion del otro
    similitudes_entre_oraciones = []
    for oracion1 in df_tokens[columna_df][i]:
        for oracion2 in df_tokens[columna_df][j]:
            dam_leven_dir_sim_ORACIONES = damerau_levenshtein_similarity_ORACIONES(oracion1, oracion2,
                                                                         tokenizacion='2gramas',
                                                                         similitud = True,
                                                                         costo_extraer=1,
                                                                         costo_insertar=1,
                                                                         costo_reemplazar=1,
                                                                         costo_intercambiar=1)
            similitudes_entre_oraciones.append(dam_leven_dir_sim_ORACIONES)
    
    # promedio
    dam_leven_dir_sim_PROMEDIO = statistics.mean(similitudes_entre_oraciones)
    
    # mediana aritmetica
    dam_leven_dir_sim_MEDIANA = statistics.median(similitudes_entre_oraciones)
    
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim_PROMEDIO, dam_leven_dir_sim_MEDIANA])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_2gramas_indirecto_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_INDIRECTO_2gramas_PROMEDIO', 'Similitud_Damerau_Levenshtein_INDIRECTO_2gramas_MEDIANA'])

'''



# 2)
# funcion para damerau-levenshtein entre textos enteros tokenizados
def damerau_levenshtein_similarity_TOKENS(tokens_1, tokens_2, similitud, costo_extraer, costo_insertar, costo_reemplazar, costo_intercambiar):
    
    dam_lev_sim = damerauLevenshtein(tokens_1, tokens_2,
                                     similarity=similitud,
                                     deleteWeight = costo_extraer,
                                     insertWeight = costo_insertar,
                                     replaceWeight = costo_reemplazar, 
                                     swapWeight = costo_intercambiar)
    
    return dam_lev_sim

# insertar palabra
damerau_levenshtein_similarity_TOKENS(['esto', 'es', 'una', 'prueba'], ['esto', 'no', 'es', 'una', 'prueba'],
                                         similitud = True,
                                         costo_extraer=1,
                                         costo_insertar=1,
                                         costo_reemplazar=1,
                                         costo_intercambiar=1)

# reemplazar palabra
damerau_levenshtein_similarity_TOKENS(['salio', 'bien'], ['salio', 'mal'],
                                         similitud = True,
                                         costo_extraer=1,
                                         costo_insertar=1,
                                         costo_reemplazar=1,
                                         costo_intercambiar=1)

# transponer e insertar palabras
damerau_levenshtein_similarity_TOKENS(['salio', 'bien'], ['que', 'bien', 'salio'],
                                         similitud = True,
                                         costo_extraer=1,
                                         costo_insertar=1,
                                         costo_reemplazar=1,
                                         costo_intercambiar=1)

# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens palabras, ponderaciones iguales
columna_df = 'Tokens_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=1,
                                                       costo_insertar=1,
                                                       costo_reemplazar=1,
                                                       costo_intercambiar=1)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_palabras_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_palabras'])

# agrego la columna ID_unico, hago merge con el resto de metricas y guardo en excel
#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
nombre_base_datos = 'EA2_TP3_2020q1'
pares_dam_leven_palabras_df['ID_unico'] = nombre_base_datos + '_' + pares_dam_leven_palabras_df['ID1'].astype(str) + '_' + pares_dam_leven_palabras_df['ID2'].astype(str)
merged_df = pd.merge(merged_df, pares_dam_leven_palabras_df.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')

#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)



# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens palabras, SIN TRANSPOSICION PORQUE SE SUPONE QUE ES MAS EXIGENTE COMPUTACIONALMENTE
# EFECTIVAMENTE PARECE QUE ES MAS RAPIDO, LO CONSIDERARIAMOS
columna_df = 'Tokens_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=1,
                                                       costo_insertar=1,
                                                       costo_reemplazar=1,
                                                       costo_intercambiar=0)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_palabras_df_SinTransposicion = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_palabras_SinTransposicion'])

# agrego la columna ID_unico, hago merge con el resto de metricas y guardo en excel
#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
nombre_base_datos = 'EA2_TP3_2020q1'
pares_dam_leven_palabras_df_SinTransposicion['ID_unico'] = nombre_base_datos + '_' + pares_dam_leven_palabras_df_SinTransposicion['ID1'].astype(str) + '_' + pares_dam_leven_palabras_df_SinTransposicion['ID2'].astype(str)
merged_df = pd.merge(merged_df, pares_dam_leven_palabras_df_SinTransposicion.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')

#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)



'''
# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens oraciones, ponderaciones iguales
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=1,
                                                       costo_insertar=1,
                                                       costo_reemplazar=1,
                                                       costo_intercambiar=1)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_oraciones_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_oraciones'])
'''

''' este de ngramas no lo haria porque son ngramas superpuestos
# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens 3gramas, ponderaciones iguales
columna_df = 'Tokens_3gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=1,
                                                       costo_insertar=1,
                                                       costo_reemplazar=1,
                                                       costo_intercambiar=1)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_3gramas_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_3gramas'])
'''
'''
# calculo lo mismo pero esta vez PONDERANDO QUE SEA MAS COSTOSO LA EXTRACCION O INSERCION, CON TAL DE QUE SI SUCEDE ESTO, SEAN MENOS SIMIL LOS TEXTOS, y hacer nula la ponderacion de reemplazo e intercambio
# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens palabras, ponderado
columna_df = 'Tokens_palabras'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=1,
                                                       costo_insertar=1,
                                                       costo_reemplazar=0,
                                                       costo_intercambiar=0)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_palabras_ponderado_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_palabras_PONDERADO'])
'''
'''
# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens oraciones, ponderado
columna_df = 'Tokens_oraciones'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=2,
                                                       costo_insertar=2,
                                                       costo_reemplazar=0,
                                                       costo_intercambiar=0)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_oraciones_ponderado_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_oraciones_PONDERADO'])
'''

'''
# Calculo los pares de similitudes de Dam-Leven directo entre textos, tokens 3gramas, ponderado
columna_df = 'Tokens_3gramas'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    dam_leven_dir_sim = damerau_levenshtein_similarity_TOKENS(df_tokens[columna_df][i], df_tokens[columna_df][j],
                                                       similitud = True,
                                                       costo_extraer=2,
                                                       costo_insertar=2,
                                                       costo_reemplazar=1,
                                                       costo_intercambiar=1)
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], dam_leven_dir_sim])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_dam_leven_3gramas_ponderado_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Damerau_Levenshtein_DIRECTO_3gramas_PONDERADO'])
'''

from itertools import combinations

# importo las metricas para no tener que volver a calcularlas
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx")
merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx")


################################ Longest Common Subsequence (la subsecuencia no tiene que ser continua)

import pylcs

df_tokens.columns

def similitud_longest_common_subsequence(texto1, texto2):
    
    # calculo la lcssequence
    lcss_distancia = pylcs.lcs_sequence_length(texto1, texto2)
    
    # calculo el texto de menor largo, que es el maximo valor que podria tomar lcss
    if len(texto1)<=len(texto2):
        largo_min = len(texto1)
    else:
        largo_min = len(texto2)
    
    # normalizo la metrica
    lcss_similitud = (lcss_distancia/largo_min)
    
    return lcss_similitud

# Calculo los pares de similitudes con LCSSec
columna_df = 'Texto'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    lcss = similitud_longest_common_subsequence(df_tokens[columna_df][i], df_tokens[columna_df][j])
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], lcss])
    print(i, j)

# Crear un DataFrame con la matriz de similitud
pares_LCSSecquence_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_LCSSecquence'])

# agrego la columna ID_unico, hago merge con el resto de metricas y guardo en excel
#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
nombre_base_datos = 'EA2_TP3_2020q1'
pares_LCSSecquence_df['ID_unico'] = nombre_base_datos + '_' + pares_LCSSecquence_df['ID1'].astype(str) + '_' + pares_LCSSecquence_df['ID2'].astype(str)
merged_df = pd.merge(merged_df, pares_LCSSecquence_df.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')

#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)

# importo las metricas para no tener que volver a calcularlas
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx")


################################ Longest Common Substring (la subsecuencia sí tiene que ser continua)

def similitud_longest_common_substring(texto1, texto2):
    
    # calculo la lcsstring
    lcss_distancia = pylcs.lcs_string_length(texto1, texto2)
    
    # calculo el texto de menor largo, que es el maximo valor que podria tomar lcss
    if len(texto1)<=len(texto2):
        largo_min = len(texto1)
    else:
        largo_min = len(texto2)
    
    # normalizo la metrica
    lcss_similitud = (lcss_distancia/largo_min)
    
    return lcss_similitud

# Calculo los pares de similitudes con LCSString
columna_df = 'Texto'
num_textos = len(df_tokens)
similarity_list = []
for i, j in combinations(range(num_textos), 2):
    lcss = similitud_longest_common_substring(df_tokens[columna_df][i], df_tokens[columna_df][j])
    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], lcss])
    print(i, j)

#pd.DataFrame({'intermedio':similarity_list}).to_excel('guardado_intermedio.xlsx')


# Crear un DataFrame con la matriz de similitud
pares_LCSString_df = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_LCSString'])

# agrego la columna ID_unico, hago merge con el resto de metricas y guardo en excel
#nombre_base_datos = 'EA2_TP1_2020q1'
#nombre_base_datos = 'EA2_TP2_2020q1'
nombre_base_datos = 'EA2_TP3_2020q1'
pares_LCSString_df['ID_unico'] = nombre_base_datos + '_' + pares_LCSString_df['ID1'].astype(str) + '_' + pares_LCSString_df['ID2'].astype(str)
merged_df = pd.merge(merged_df, pares_LCSString_df.drop(['ID1', 'ID2'], axis=1), on=['ID_unico'], how='outer')

#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)


# importo las metricas para no tener que volver a calcularlas
merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx")
#merged_df = pd.read_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx")

import pandas as pd
merged_df['Similitud_LCSSecquence'] = 1 - merged_df['Similitud_LCSSecquence']
merged_df['Similitud_LCSString'] = 1 - merged_df['Similitud_LCSString']

merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP1_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP2_2020q1.xlsx", index=False)
#merged_df.to_excel("Métricas Calculadas para Similitud/metricas_similitudes_EA2_TP3_2020q1.xlsx", index=False)




