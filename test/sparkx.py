import numpy as np
import pandas as pd
import scanpy as sc
import anndata

numCores=4
option='mixture'

example_data = '../examples_Python_version/ChickenHeart.h5ad'
example_adata = anndata.read_h5ad(example_data)

def sparkx(count, location, genenames, option='mixture', num_cores=1):
    """Run SPAKR-X python version
    Parameters
    ----------
    count : scipy.sparse matrix or np.ndarray
        Gene expression count matrix (cells x genes).
    location : np.ndarray
        Spatial coordinates matrix (cells x dimensions).
    genenames : np.ndarray
        Array of gene names corresponding to count columns. 
    option : str, optional
        Type of kernel to use ('mixture' or 'projection'). Default is 'mixture'.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. Default is 1.
    """
    assert count.shape[1] == len(genenames), "Length of genenames must match number of columns in count"
    from statsmodels.stats.multitest import multipletests
    # --- perform filtering on cells / genes
    totalcount = count.sum(axis=1)
    keep_cell_idx = np.where(totalcount != 0)[0]
    count = count[keep_cell_idx, :]
    location = location[keep_cell_idx, :]
    genecount = np.array(count.sum(axis=0)).ravel()
    keep_gene_idx = np.where(genecount != 0)[0]
    count = count[:, keep_gene_idx]
    genenames = genenames[keep_gene_idx]
    if np.sum(pd.isna(genenames)) > 0:
        genenames[pd.isna(genenames)] = "NAgene"
    numGene = count.shape[1]
    numCell = count.shape[0]
    # --- run SPARKX ---
    print("## ===== SPARK-X INPUT INFORMATION ==== ")
    print(f"## number of total samples: {numCell}")
    print(f"## number of total genes: {numGene}")
    if num_cores > 1:
        print(f"## Running with {num_cores} cores")
    else:
        print("## Running with single core, may take some time")
    sparkx_list = list()
    print("## Testing With Projection Kernel")
    final_location = location
    sparkx_res = sparkx_sk(count, final_location, num_cores=num_cores)
    sparkx_res.index = genenames
    sparkx_list.append(sparkx_res)
    if option == 'mixture':
        # --- record run time
        import time
        start_time = time.time()
        for iker in range(5):
            print(f"## Testing With Gaussian Kernel {iker+1}")
            final_location = transloc_func_vec(location, lker=iker, transfunc="gaussian")
            sparkx_list.append(sparkx_sk(count, final_location, num_cores=num_cores))
        for iker in range(5):
            print(f"## Testing With Cosine Kernel {iker+1}")
            final_location = transloc_func_vec(location, lker=iker, transfunc="cosine")
            sparkx_list.append(sparkx_sk(count, final_location, num_cores=num_cores))
        end_time = time.time()
        print(f"Time taken for mixture kernels: {end_time - start_time:.2f} seconds")
    # --- combine p-values from different kernels using ACAT ---
    allstat = np.column_stack([x["stat"] for x in sparkx_list])
    allpvals = np.column_stack([x["pval"] for x in sparkx_list])
    allstat = pd.DataFrame(allstat, index=genenames)    # shape: n_genes × n_methods
    allpvals = pd.DataFrame(allpvals, index=genenames)
    comb_pval = allpvals.apply(ACAT, axis=1)
    pBY = multipletests(comb_pval, method="fdr_by")[1]
    res_sparkx = pd.DataFrame({
        'combinedPval': comb_pval,
        'adjustedPval': pBY
    }, index=genenames)
    res_sparkx = res_sparkx.sort_values(by='adjustedPval')
    significant_gene_number = (res_sparkx['adjustedPval'] <= 0.05).sum()
    significant_gene_number = min(res_sparkx.shape[0], max(significant_gene_number, 500))  # avoid too few genes
    SVGs = res_sparkx.sort_values('adjustedPval').iloc[:significant_gene_number, :]
    return SVGs


def cluster_SVGs(SVG_mat, k=10):
    # --- obtain result from Scanpy
    SVG_adata = sc.AnnData(X=SVG_mat)
    sc.tl.pca(SVG_adata)
    sc.pp.neighbors(SVG_adata, n_neighbors=100, n_pcs=30)
    resolution = 1.0
    sc.tl.louvain(SVG_adata, 
                 resolution=resolution, 
                 key_added=f"louvain_{resolution}",
                 use_weights=True)
    while len(set(SVG_adata.obs[f"louvain_{resolution}"])) < k:
        resolution += 0.1
        sc.tl.louvain(SVG_adata, 
                    resolution=resolution, 
                    key_added=f"louvain_{resolution}",
                    use_weights=True)
    return SVG_adata.obs[f"louvain_{resolution}"].values

    
def sparkx_sk(counts, infomat, num_cores=1):
    """Simplified SPARK-X without covariate matrix
    """
    from multiprocessing import Pool
    Xinfomat = infomat - infomat.mean(axis=0, keepdims=True)
    XtX = Xinfomat.T @ Xinfomat
    loc_inv = np.linalg.inv(XtX)
    kmat_first = Xinfomat @ loc_inv
    Klam = np.linalg.eigvalsh(Xinfomat.T @ kmat_first) # guarantees real eigenvalues if the matrix is close to symmetric.
    EHL = counts.T @ Xinfomat
    numCell = Xinfomat.shape[0]
    counts_squared = counts.power(2)
    adjust_nominator = np.array(counts_squared.sum(axis=0)).ravel()
    vec_stat = np.einsum('ij,jk,ik->i', EHL, loc_inv, EHL)  # shape: (n_rows,)
    # scale by numCell and divide by adjust_nominator
    vec_stat = vec_stat * numCell / adjust_nominator
    vec_ybar = np.array(counts.mean(axis=0)).ravel()
    vec_ylam = 1 - numCell * vec_ybar**2 / adjust_nominator  # vectorized
    # Compute davies p-value in parallel ---
    with Pool(num_cores) as pool:
        vec_daviesp = pool.starmap(
            sparkx_pval,
            [(i, vec_ylam, Klam, vec_stat) for i in range(counts.shape[1])]
        )    
    vec_daviesp = np.array(vec_daviesp)  # convert to NumPy array
    res_sparkx = pd.DataFrame({
        'stat': vec_stat,
        'pval': vec_daviesp
    })
    return res_sparkx



def sparkx_pval(igene,lambda_G,lambda_K,allstat):
    from chi2comb import chi2comb_cdf, ChiSquared # implementation of davies function in r: https://www.rdocumentation.org/packages/CompQuadForm/versions/1.4.4/topics/davies
    try:
        # sort lambda_G[igene] * lambda_K in decreasing order
        Zsort = np.sort(lambda_G[igene] * lambda_K)[::-1]
        gcoef = 0
        dofs = [1.0] * len(Zsort)
        ncents = [0.0] * len(Zsort)
        q = allstat[igene]
        chi2s = [ChiSquared(Zsort[i], ncents[i], dofs[i]) for i in range(len(Zsort))]
        result, errno, info = chi2comb_cdf(q, chi2s, gcoef)
        pout = 1 - result
        if result <= 0 or result >= 1.0:
            pout = liu(allstat[igene], Zsort)
    except Exception:
        pout = liu(allstat[igene], Zsort)
    return pout


def liu(q, lambdas, h=None, delta=None):
    """ -> converted by ChatGPT, original code from https://github.com/cran/CompQuadForm/blob/master/R/liu.R
    Liu approximation for the distribution of quadratic forms
    in normal variables (R 'liu' function).

    Parameters
    ----------
    q : float
        The value to evaluate P[Q > q]
    lambdas : array-like
        Eigenvalues (lambda)
    h : array-like, optional
        Multiplicities (default 1 for each lambda)
    delta : array-like, optional
        Non-centrality parameters (default 0 for each lambda)

    Returns
    -------
    float
        P-value P[Q > q]
    """
    from scipy.stats import ncx2
    lambdas = np.asarray(lambdas)
    r = len(lambdas)
    if h is None:
        h = np.ones(r)
    else:
        h = np.asarray(h)
        if len(h) != r:
            raise ValueError("lambda and h should have the same length!")
    if delta is None:
        delta = np.zeros(r)
    else:
        delta = np.asarray(delta)
        if len(delta) != r:
            raise ValueError("lambda and delta should have the same length!")
        if np.any(delta < 0):
            raise ValueError("All non centrality parameters in 'delta' should be positive!")
    c1 = np.sum(lambdas * h) + np.sum(lambdas * delta)
    c2 = np.sum(lambdas**2 * h) + 2 * np.sum(lambdas**2 * delta)
    c3 = np.sum(lambdas**3 * h) + 3 * np.sum(lambdas**3 * delta)
    c4 = np.sum(lambdas**4 * h) + 4 * np.sum(lambdas**4 * delta)
    s1 = c3 / (c2 ** (3 / 2))
    s2 = c4 / (c2 ** 2)
    muQ = c1
    sigmaQ = np.sqrt(2 * c2)
    tstar = (q - muQ) / sigmaQ
    if s1**2 > s2:
        a = 1 / (s1 - np.sqrt(s1**2 - s2))
        delta = s1 * a**3 - a**2
        l = a**2 - 2 * delta
    else:
        a = 1 / s1
        delta = 0
        l = c2**3 / c3**2
    muX = l + delta
    sigmaX = np.sqrt(2) * a
    # Survival function (upper tail)
    try:
        Qq = ncx2.sf(tstar * sigmaX + muX, df=l, nc=delta) 
    except Exception:
        print("Error in computing survival function")
    return Qq

def ACAT(pvals, weights=None):
    # --- converted by ChatGPT, original code from: https://github.com/xzhoulab/SPARK/blob/a8b4bf27b804604dfda53da42992f100b8e4e727/R/sparkx.R#L307
    from scipy.stats import cauchy
    import warnings
    # check for NAs
    if np.any(np.isnan(pvals)):
        raise ValueError("Cannot have NAs in the p-values!")
    # check if pvals are between 0 and 1
    if np.any(pvals < 0) or np.any(pvals > 1):
        raise ValueError("P-values must be between 0 and 1!")
    # check if there are pvals that are exactly 0 or 1
    is_zero = np.any(pvals == 0)
    is_one = np.any(pvals == 1)
    if is_zero and is_one:
        raise ValueError("Cannot have both 0 and 1 p-values!")
    if is_zero:
        return 0.0
    if is_one:
        warnings.warn("There are p-values that are exactly 1!")
        return 1.0
    # default equal weights, or normalize user-supplied weights
    n = len(pvals)
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.array(weights)
        if len(weights) != n:
            raise ValueError("The length of weights should be the same as that of the p-values")
        if np.any(weights < 0):
            raise ValueError("All the weights must be positive!")
        weights = weights / np.sum(weights)
    # handle very small p-values
    is_small = pvals < 1e-16
    if not np.any(is_small):
        cct_stat = np.sum(weights * np.tan((0.5 - pvals) * np.pi))
    else:
        cct_stat = np.sum(weights[is_small] / (np.pi * pvals[is_small]))
        cct_stat += np.sum(weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi))
    # compute p-value
    if cct_stat > 1e15:
        pval = 1 / (cct_stat * np.pi)
    else:
        pval = 1 - cauchy.cdf(cct_stat)
    return pval

def transloc_func_vec(coord, lker, transfunc="gaussian"):
    # center each column
    coord = coord - np.mean(coord, axis=0)
    # compute quantiles per column
    probs = np.arange(0.2, 1.01, 0.2)
    # l will be a 2D array: each column has its quantiles
    l = np.quantile(np.abs(coord), q=probs, axis=0)
    if transfunc == "gaussian":
        out = np.exp(-coord**2 / (2 * l[lker, :][np.newaxis, :]**2))
    elif transfunc == "cosine":
        out = np.cos(2 * np.pi * coord / l[lker, :][np.newaxis, :])
    else:
        raise ValueError("transfunc must be 'gaussian' or 'cosine'")
    return out


import sys
sys.path.append('/net/mulan/home/wenjinma/projects/SpaDOT/SpaDOT/utils/sctransform')
from sctransform import SCTransform

timepoints = example_adata.obs['timepoint'].unique()
for tp in timepoints:
    tp_adata = example_adata[example_adata.obs['timepoint'] == tp]
    tp_adata.layers['counts'] = tp_adata.X.copy()
    # --- try sctransform implemented by Stereopy
    assay_out, vst_out = SCTransform(tp_adata.X.T,
                            genes=tp_adata.var_names,
                            cells=tp_adata.obs_names,
                            return_only_var_genes=False, 
                            n_cells=5000,
                            variable_features_n=None,
                            variable_features_rv_th=1.3)
    tp_adata = tp_adata[:, assay_out['scale.data'].index]
    print(f'Timepoint: {tp}, Number of cells: {tp_adata.n_obs}, Number of genes: {tp_adata.n_vars}')
    count_spark = tp_adata.layers['counts']
    locations_spark = tp_adata.obsm['spatial']
    # --- use scanpy transform to substitute SCTransform: skip for now and check results later
    SVGs = sparkx(count_spark, locations_spark, np.array(tp_adata.var_names), option=option, num_cores=numCores)
    SVGs.to_csv(f'./{tp}_SVG_sparkx.csv')
    SVG_clusters = cluster_SVGs(assay_out['scale.data'].loc[SVGs.index, :], k=10)
    SVGs['cluster'] = SVG_clusters
    SVGs.to_csv(f'./{tp}_SVG_sparkx_clustered_louvain.csv')