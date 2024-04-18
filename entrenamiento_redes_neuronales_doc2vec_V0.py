# ENTRENAMIENTO DE REDES NEURONALES PARA WORD EMBEDDINGS

################ Vectorización Doc2Vec


#import pandas as pd
#import gensim
#from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
#import os
import logging # configuro los mensajes de los logs
#import nltk
import json
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# configuro los logs
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

##### traigo datos preprocesados

# EA2_TP1_2020q1
with open(r"C:\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\palabras_EA2_TP1_2020q1.json", "r") as archivo:
    palabras_EA2_TP1_2020q1 = json.load(archivo)

# EA2_TP2_2020q1
with open(r"C:\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\palabras_EA2_TP2_2020q1.json", "r") as archivo:
    palabras_EA2_TP2_2020q1 = json.load(archivo)

# EA2_TP3_2020q1
with open(r"C:\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\palabras_EA2_TP3_2020q1.json", "r") as archivo:
    palabras_EA2_TP3_2020q1 = json.load(archivo)

##### uno los textos en una sola lista
documentos = palabras_EA2_TP1_2020q1 + palabras_EA2_TP2_2020q1 + palabras_EA2_TP3_2020q1 

# convierto los documentos en objetos TaggedDocument
documentos_etiquetados = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documentos)]

# creo el modelo de Doc2Vec
modelo = Doc2Vec(vector_size=50,
                 window=2,
                 min_count=6,
                 dm=0,
                 epochs=40)

# entreno el modelo
modelo.build_vocab(documentos_etiquetados)
modelo.train(documentos_etiquetados, total_examples=modelo.corpus_count, epochs=modelo.epochs)

# guardo el modelo
modelo.save("C:\\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\modelo_doc2vec_informes_vectors50_window2_minc2_dm0_epochs40.model")

# palabras en el modelo
modelo.wv.index_to_key

# cantidad de palabras unicas en el modelo
len(modelo.wv.index_to_key)


################## validacion del modelo

# pruebas de vectores inferidos simples
vector1 = modelo.infer_vector(['rechazar', 'hipotesis', 'nula'])
vector2 = modelo.infer_vector(['hipotesis', 'nula', 'ir', 'rechazar'])
vector3 = modelo.infer_vector(['explicar', 'hipotesis'])

cosine_similarity(vector1.reshape(1,-1), vector2.reshape(1,-1))[0][0]
cosine_similarity(vector1.reshape(1,-1), vector3.reshape(1,-1))[0][0]

# prueba avanzada

ranks = []
second_ranks = []
for doc_id in range(len(documentos_etiquetados)):
    inferred_vector = modelo.infer_vector(documentos_etiquetados[doc_id].words)
    sims = modelo.dv.most_similar([inferred_vector], topn=len(modelo.dv))
    rank = [docid for docid, sim in sims].index(str(doc_id))
    ranks.append(rank)
    second_ranks.append(sims[1])


# Pick a random document from the corpus and infer a vector from the model
import random
doc_id = random.randint(0, len(documentos_etiquetados) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(documentos_etiquetados[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(int(sim_id[0]), ' '.join(documentos_etiquetados[int(sim_id[0])].words)))

