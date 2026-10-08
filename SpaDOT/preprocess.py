import os
import random
import numpy as np
import anndata
from SpaDOT.utils import _preprocess_utils

seed=1993
random.seed(seed)

def preprocess(args):
    '''
    Preprocess the data for SpaDOT model.
    '''
    data_dir = os.path.abspath(args.data)
    if not args.output_dir:
        args.output_dir = os.path.dirname(data_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    adata = anndata.read_h5ad(data_dir)
    # check if `timepoint` in adata.obs
    if 'timepoint' not in adata.obs.columns:
        raise ValueError("The `timepoint` column is not found in adata.obs. Please make sure timepoint information is given.")
    if 'spatial' not in adata.obsm.keys():
        raise ValueError("The `spatial` key is not found in adata.obsm. Please make sure spatial coordinates are provided.")
    # check if adata.obsm['spatial'] is 2D numpy array
    if not isinstance(adata.obsm['spatial'], np.ndarray) or adata.obsm['spatial'].ndim != 2:
        raise ValueError("The `spatial` key in adata.obsm is not a 2D numpy array. Please make sure spatial coordinates are correctly provided.")
    # preprocess adata based on whether performing SVG selection
    preprocessed_adata = _preprocess_utils.preprocess_adata(args, adata)
    preprocessed_adata.write_h5ad(os.path.join(args.output_dir, args.prefix + os.path.basename(data_dir)))
