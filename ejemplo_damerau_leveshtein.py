# Ejemplo de uso similitud/distancia de Damereu-Levenshtein

from fastDamerauLevenshtein import damerauLevenshtein
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ngrams

# EJEMPLOS CON TOKENIZACIONES DE PALABRAS
# oración 1: Rechazo la hipótesis nula
# oración 2: No rechazo la hipótesis nula

damerauLevenshtein(['rechazo', 'la', 'hipotesis', 'nula'], ['no', 'rechazo', 'la', 'hipotesis', 'nula'], similarity=False)
damerauLevenshtein(['rechazo', 'la', 'hipotesis', 'nula'], ['no', 'rechazo', 'la', 'hipotesis', 'nula'], similarity=True)

damerauLevenshtein(['rechazo', 'la', 'hipotesis', 'nula'], ['no', 'rechazo', 'la', 'hipotesis', 'nula'],
                   similarity=False,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1, 
                   swapWeight = 1) 
damerauLevenshtein(['rechazo', 'la', 'hipotesis', 'nula'], ['no', 'rechazo', 'la', 'hipotesis', 'nula'],
                   similarity=True,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1, 
                   swapWeight = 1)



# ejemplo de uso en caso de LEMATIZAR las palabras y remover stopwords

# oración 1: Rechazó la hipótesis nula
# oración 2: La hipótesis nula fue rechazada



damerauLevenshtein(['rechazar', 'hipotesis', 'nula'], ['hipotesis', 'nula', 'fue', 'rechazar'],
                   similarity=False,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1,
                   swapWeight = 1)
damerauLevenshtein(['rechazar', 'hipotesis', 'nula'], ['hipotesis', 'nula', 'fue', 'rechazar'],
                   similarity=True,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1, 
                   swapWeight = 1)

# ejemplo de uso en caso de LEMATIZAR las palabras y remover stopwords, CON PONDERACIONES DE COSTO DE OPERACIONES
damerauLevenshtein(['rechazar', 'hipotesis', 'nula'], ['hipotesis', 'nula', 'fue', 'rechazar'],
                   similarity=False,
                   deleteWeight = 2, # le doy mas peso al costo de extraer para que sean iguales (si tengo que extraer, son menos iguales)
                   insertWeight = 2, # le doy mas peso al costo de insertar para que sean iguales (si tengo que insertar, son menos iguales)
                   replaceWeight = 1, 
                   swapWeight = 1) 
damerauLevenshtein(['rechazar', 'hipotesis', 'nula'], ['hipotesis', 'nula', 'fue', 'rechazar'],
                   similarity=True,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 2, 
                   swapWeight = 2)


# EJEMPLOS CON TOKENIZACIONES DE 2GRAMAS
# ejemplo de uso en caso de LEMATIZAR las palabras y remover stopwords
damerauLevenshtein(['rechazar hipotesis', ' hipotesis nula'], ['hipotesis nula', 'nula fue', ' fue rechazar'],
                   similarity=False,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1,
                   swapWeight = 1)
damerauLevenshtein(['rechazar hipotesis', ' hipotesis nula'], ['hipotesis nula', 'nula fue', ' fue rechazar'],
                   similarity=True,
                   deleteWeight = 1,
                   insertWeight = 1,
                   replaceWeight = 1, 
                   swapWeight = 1)



