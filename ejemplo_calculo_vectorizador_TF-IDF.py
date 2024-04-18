from sklearn.feature_extraction.text import TfidfVectorizer

# Definir las frases
frases = ["ser basar metodo estadistica tomar decision", "ser metodo fantastico ser tomar decision"]

# Crear un objeto TfidfVectorizer
vectorizador = TfidfVectorizer()

# Ajustar y transformar las frases para calcular los vectores TF-IDF
vectores_tfidf = vectorizador.fit_transform(frases)

# Obtener el vocabulario
vocabulario = vectorizador.get_feature_names_out()

# Mostrar los vectores TF-IDF y el vocabulario
print("Vectores TF-IDF:")
print(vectores_tfidf.toarray())
print("Vocabulario:")
print(vocabulario)
