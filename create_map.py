# 1️⃣ Instalar librerías si no las tenés
# pip install sentence-transformers umap-learn matplotlib

from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt

# 2️⃣ Lista de palabras que querés mapear
words = ["perro", "canino", "gato", "auto", "vehículo", "comida", "manzana", "avión"]

# 3️⃣ Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4️⃣ Obtener embeddings
embeddings = model.encode(words)  # devuelve un vector 384D por palabra

# 5️⃣ Reducir dimensiones a 2D
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# 6️⃣ Graficar
plt.figure(figsize=(8,6))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])

for i, word in enumerate(words):
    plt.text(embeddings_2d[i,0]+0.01, embeddings_2d[i,1]+0.01, word, fontsize=12)

plt.title("Mapa 2D de palabras por significado")
plt.show()
