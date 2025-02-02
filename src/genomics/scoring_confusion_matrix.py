import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def get_cancer_score(search_embedding, reference_embeddings, labels, k=100):
    if len(search_embedding.shape) == 1:
        search_embedding = search_embedding.reshape(1, -1)
    labels = labels.astype(int)

    similarities = cosine_similarity(search_embedding, reference_embeddings)[0]
    # print(np.max(similarities))
    top_k_indices = np.argsort(similarities)[::-1][:k]
    # print(similarities[top_k_indices[0]])
    top_k_labels = labels[top_k_indices]
    # cancer_score = np.mean(top_k_labels)
    label_counts = Counter(top_k_labels)
    # print(label_counts)
    # input()
    # # MOST COMMON
    # most_common_label = label_counts.most_common(1)[0][0]
    # return most_common_label
    # probabilities = np.zeros(12)
    # for label in top_k_labels:
    #     probabilities[label - 1] += 1
    #
    # # Normalize to get probabilities
    # probabilities /= k
    # return probabilities

    count_1 = label_counts.get(1, 0)

    threshold = 30
    return 1 if count_1 >= threshold else 0

reference_embeddings = np.load('cancer_reference_embeddings_finetune.npy')
labels = np.load('label.npy')

# search_embedding = np.random.normal(size=512)
# # print(search_embedding)
# score = get_cancer_score(search_embedding, reference_embeddings, labels)
# print(score)


import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path
import warnings
import os
import tempfile

# tempfile.tempdir = 'path/to/writable/directory'
import scanpy as sc
import scib
import numpy as np
import sys
sys.path.insert(0, "../")

import scgpt as scg
import matplotlib.pyplot as plt
import anndata
import pandas as pd
FINETUNE = True
# NAME = 'E13_integrated_subsetneuroblast'
NAME = 'cancer_test'
# NAME = 'P21_test'
MODEL = 'scGPT_cancer'


warnings.simplefilter("ignore", ResourceWarning)

model_dir = Path(f"../{MODEL}")
mode = "cls"

sample_data_path = rf"../../data/{NAME}.h5ad"
adata = sc.read_h5ad(sample_data_path)



gene_col = "feature_name"
adata.var[gene_col] = adata.var[gene_col].str.upper()
# adata.var[gene_col] = adata.var.index.str.upper()
cell_type_key = "disease"
batch_key = "cell_type"
N_HVG = 3000
org_adata = adata.copy()
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]
embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=1,
    max_length=N_HVG+1
)
predictions = []
ground_truth = []
for i, search_embedding in enumerate(embed_adata.obsm["X_scGPT"]):
    prediction = get_cancer_score(search_embedding, reference_embeddings, labels)
    # prediction = [org_adata.obs['predicted.id'].values[i]] + probabilities.tolist()
    if prediction == 1:

        predictions.append('breast cancer')
    elif prediction == 0:
        predictions.append('normal')

    ground_truth.append((embed_adata.obs[cell_type_key].values[i]))

labels = sorted(list(set(ground_truth + predictions)))  # Get unique sorted labels
conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)

# Convert to DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix,
                             index=[f"{label}" for label in labels],
                             columns=[f"{label}" for label in labels])
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for the plots
sns.set(style='whitegrid')

# Create and save the heatmap for the confusion mat rix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues',
            xticklabels=conf_matrix_df.columns, yticklabels=conf_matrix_df.index)
plt.title('Confusion Matrix P21')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig(rf'confusion_matrix_{NAME}_threshold30.png', dpi=300)


# Set the aesthetics for the plots
sns.set(style='whitegrid')

total_samples = conf_matrix.sum(axis=1, keepdims=True)
percentage_conf_matrix = (conf_matrix / total_samples) * 100

# Handle division by zero for classes with no samples
percentage_conf_matrix = np.nan_to_num(percentage_conf_matrix, nan=0.0)

# Convert to DataFrame
percentage_conf_matrix_df = pd.DataFrame(
    percentage_conf_matrix,
    index=[f"{label}" for label in labels],
    columns=[f"{label}" for label in labels]
).round(2)  # Round to 2 decimal places

# Then modify your visualization code:
plt.figure(figsize=(12, 10))
sns.heatmap(percentage_conf_matrix_df, annot=True, fmt=".2f", cmap='Blues',
            cbar_kws={'label': 'Percentage (%)'},
            xticklabels=percentage_conf_matrix_df.columns,
            yticklabels=percentage_conf_matrix_df.index)
plt.title('Confusion Matrix P21 (Percentage)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# Save with new filename
plt.savefig(rf'confusion_matrix_percentage_{NAME}_threshold30.png', dpi=300, bbox_inches='tight')