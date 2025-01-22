print('0')
import os

from pathlib import Path
import warnings
import os
import tempfile

# tempfile.tempdir = 'path/to/writable/directory'
import scanpy as sc
import scib
import numpy as np
import sys
print('1')
sys.path.insert(0, "../")

import scgpt as scg
import matplotlib.pyplot as plt
import anndata
FINETUNE = False
NAME = 'cancer'
plt.style.context('default')
warnings.simplefilter("ignore", ResourceWarning)

model_dir = Path("../scGPT_human")
mode = "cls" # "pad" for genes; "cls" for cell
print('2')
# Evaluation
"""
Calculate the metrics for integration results
"""

sample_data_path = rf"../../data/{NAME}.h5ad"
adata = sc.read_h5ad(sample_data_path)
mask = [x in ['normal', 'breast cancer'] for x in adata.obs['disease']]

adata = adata[mask]

# # Generate random indices
# random_indices = np.random.choice(adata.n_obs, size=1000, replace=False)
#
# # Subset the AnnData object
# adata = adata[random_indices, :]

# adata.write_h5ad(sample_data_path)
print('3')
gene_col = "feature_name"
adata.var[gene_col] = adata.var[gene_col].str.upper()
print(adata.var)
cell_type_key = "disease"
batch_key = "cell_type"
N_HVG = 3000
org_adata = adata.copy()
# celltype_id_labels = adata.obs[cell_type_key].astype("category").cat.codes.values
# adata = adata[celltype_id_labels >= 0]
print('4.1')
# adata.layers['counts'] = adata.X.copy()
# sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print('4.2')
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]
print('5')
embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=1,
    max_length=N_HVG+1
)
print('6')
if mode == "cls":
    sc.pp.neighbors(embed_adata, use_rep="X_scGPT")
    sc.tl.umap(embed_adata)
    if FINETUNE:
        sc.pl.umap(embed_adata,
                   color=[cell_type_key, batch_key],
                   frameon=False,
                   wspace=0.4,
                   title=[f"scGPT finetune: {cell_type_key}", f"scGPT finetune: {batch_key}"],
                    )
        plt.savefig(f"figures/scGPT_finetune_{NAME}.png")
    else:
        sc.pl.umap(embed_adata,
                   color=[cell_type_key, batch_key],
                   frameon=False,
                   wspace=0.4,
                   title=[f"scGPT zeroshot: {cell_type_key}", f"scGPT zeroshot: {batch_key}"],
                   )
        plt.savefig(f"figures/scGPT_zeroshot_{NAME}.png")