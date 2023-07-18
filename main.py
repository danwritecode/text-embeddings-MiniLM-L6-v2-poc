from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random


# Generate dayas
dayas = []

for r in range(0, 5000):
    daya = []
    daya.append(random.randint(0, 1000)) # user id

    months = []
    for mr in range(0, 11):
        months.append(random.randint(0, 50000)) # month over month values

    for m in months:
        daya.append(m)

    dayas.append(daya)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(dayas)

num_clusters = 10
clustering_model = KMeans(n_clusters=num_clusters)
cluster_assignment = clustering_model.fit_predict(embeddings)
tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)

# Create a scatter plot
plt.figure(figsize=(10, 10))
for i in range(num_clusters):
    cluster_i = np.where(cluster_assignment == i)
    plt.scatter(tsne_embeddings[cluster_i, 0], tsne_embeddings[cluster_i, 1])

plt.show()
