import scanpy as sc
import numpy as np

# Load the data
sample_data_path = r"../../data/cancer.h5ad"
adata = sc.read_h5ad(sample_data_path)
mask = [x in ['normal', 'breast cancer'] for x in adata.obs['disease']]
adata = adata[mask]
# Set random seed for reproducibility
np.random.seed(42)

# Randomly select 10% of cells
n_cells = adata.n_obs
test_indices = np.random.choice(n_cells, size=int(0.1 * n_cells), replace=False)
train_indices = np.array([i for i in range(n_cells) if i not in test_indices])

# Create test and train datasets
test_adata = adata[test_indices]
train_adata = adata[train_indices]

# Save the split datasets
test_adata.write_h5ad("../../data/cancer_test.h5ad")
train_adata.write_h5ad("../../data/cancer_train.h5ad")
