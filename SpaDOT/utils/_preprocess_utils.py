import os
import scipy
import anndata
import scanpy as sc
import pandas as pd

def preprocess_adata(args, adata):
    '''
    Preprocess the data for SpaDOT model training.
    Args:
        args: arguments
        adata: AnnData object
    Returns:
        preprocessed_adata: preprocessed AnnData object
    '''
    tps = adata.obs['timepoint'].unique()
    if args.feature_selection:
        SVGs = _get_SVGs(args, tps)
        adata = adata[:, SVGs].copy()
    # save counts in adata.layers
    if not scipy.sparse.issparse(adata.X):
        adata.layers['counts'] = scipy.sparse.csr_matrix(adata.X)
    else:
        adata.layers['counts'] = adata.X
    # --- preprocess data
    tp_adata_list = []
    for tp in tps:
        tp_adata = adata[adata.obs['timepoint'] == tp]
        sc.pp.normalize_total(tp_adata, target_sum=1e-4)
        sc.pp.log1p(tp_adata)
        tp_adata_list.append(tp_adata)
    # union SVG genes for all time points
    if args.feature_selection:
        genes = set().union(*(obj.var_names for obj in tp_adata_list))
        genes = list(genes)
        genes.sort()
        with open(args.output_dir+os.sep+'SVG_genes.txt', 'w') as f:
            for item in genes:
                f.write("%s\n" % item)  
    new_tp_adata_list = []
    for tp_adata in tp_adata_list:
        if args.feature_selection:
            tp_adata = tp_adata[:, genes].copy()
        sc.pp.scale(tp_adata)
        new_tp_adata_list.append(tp_adata)
    preprocessed_adata = anndata.concat(new_tp_adata_list)
    return preprocessed_adata

def _get_SVGs(args, tps):
    '''
    Run SPARKX to select SVGs.
    Args:
        args: arguments
    Returns:
        SVG_genes: list of SVG genes
    '''
    # --- after running SPARKX, the SVGs are saved in the result directory
    ## select SVGs by cluster
    tp_SVGs = []
    for tp in tps:
        tp_SVGs_cluster = pd.read_csv(args.output_dir+os.sep+str(tp)+'_SVG_sparkx_clustered_louvain.csv', header=0, index_col=0)
        tp_SVGs.append(tp_SVGs_cluster)
    min_idx = min(range(len(tp_SVGs)), key=lambda i: len(tp_SVGs[i]))
    min_tp_SVG_len = len(tp_SVGs[min_idx])
    SVG_genes = tp_SVGs[min_idx].index.tolist()
    for idx, tp_SVG in enumerate(tp_SVGs):
        if idx == min_idx: continue
        tp_SVG_num_clusters = len(set(tp_SVG['cluster']))
        tp_SVG_top = tp_SVG.sort_values(by='adjustedPval', ascending=True).groupby('cluster').head(max(100, round(min_tp_SVG_len/tp_SVG_num_clusters)))
        SVG_genes.extend(tp_SVG_top.index.tolist())
    SVG_genes = list(set(SVG_genes))
    SVG_genes.sort()
    return SVG_genes