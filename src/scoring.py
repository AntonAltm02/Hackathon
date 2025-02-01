import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_cancer_score(search_embedding, reference_embeddings, labels, k=100):
    if len(search_embedding.shape) == 1:
        search_embedding = search_embedding.reshape(1, -1)

    similarities = cosine_similarity(search_embedding, reference_embeddings)[0]
    # print(np.max(similarities))
    top_k_indices = np.argsort(similarities)[::-1][:k]
    # print(similarities[top_k_indices[0]])
    top_k_labels = labels[top_k_indices]
    cancer_score = np.mean(top_k_labels)

    return cancer_score


reference_embeddings = np.load('reference_embeddings_finetune.npy')
labels = np.load('label.npy')

search_embedding = np.random.normal(size=512)
# print(search_embedding)
score = get_cancer_score(search_embedding, reference_embeddings, labels)
print(score)