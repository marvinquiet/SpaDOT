# --- other utilitities functions ---
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import time
import yaml
# from importlib.resources import files
import sklearn
import scipy as sp
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import random
from torch.backends import cudnn

import warnings
from scipy.stats import cauchy
from multiprocessing import Pool
from scipy.stats import ncx2
from chi2comb import chi2comb_cdf, ChiSquared # implementation of davies function in r: https://www.rdocumentation.org/packages/CompQuadForm/versions/1.4.4/topics/davies
from statsmodels.stats.multitest import multipletests

def set_seed(seed=1993):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed) # for stabilization
    # torch.use_deterministic_algorithms(True)

def seed_worker(worker_id=1993):
    np.random.seed(worker_id)
    random.seed(worker_id)

def load_model_config(args):
    '''
    Load the model configuration from the config.yaml file provided in the package.
    '''
    if args.config:
        config_path = args.config
    else:
        # Load the default config.yaml from the package
        # config_path = files("SpaDOT").joinpath("config.yaml")
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_spatial_coords(adata, x_col=None, y_col=None, csv_path=None, barcode_col=None, coord_cols=None):
    """
    Set ``adata.obsm['spatial']`` from common coordinate sources.

    SpaDOT expects spatial coordinates to be stored as a 2-D NumPy array in
    ``adata.obsm['spatial']``.  Many 10x Visium ``.h5ad`` objects produced by
    Space Ranger or loaded with ``scanpy.read_visium`` already contain this key
    and **no extra step is required**.  Call this helper only when the spatial
    information is not yet in ``adata.obsm['spatial']``.

    Three mutually exclusive sources are supported:

    1. **Columns in** ``adata.obs`` – pass *x_col* and *y_col*.
    2. **External CSV file** – pass *csv_path*.  The CSV must be indexed by cell
       barcodes (or contain a barcode column specified by *barcode_col*) and
       include two coordinate columns (optionally specified via *coord_cols*;
       defaults to the first two numeric columns).
    3. **Already in** ``adata.obsm['spatial']`` – nothing to do, the function
       returns immediately without modifying the object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to update in-place.
    x_col : str, optional
        Column name in ``adata.obs`` for the x (or row) coordinate.
    y_col : str, optional
        Column name in ``adata.obs`` for the y (or column) coordinate.
    csv_path : str, optional
        Path to a CSV file containing spatial coordinates.
    barcode_col : str, optional
        Column name in the CSV that stores cell barcodes.  When *None*, the
        CSV index is assumed to be the barcode.
    coord_cols : list of str, optional
        Two column names in the CSV to use as [x, y] coordinates.  When
        *None*, the first two numeric columns are selected automatically.

    Returns
    -------
    adata : anndata.AnnData
        The same object, modified in-place.

    Examples
    --------
    **Case 1 – 10x Visium  already has adata.obsm['spatial']**

    **Case 2 – coordinates stored in** ``adata.obs``
    >>> set_spatial_coords(adata, x_col="pixel_x", y_col="pixel_y")

    **Case 3 – coordinates in an external CSV**
    >>> set_spatial_coords(adata, csv_path="tissue_positions_list.csv",
    ...                    coord_cols=["pixel_x", "pixel_y"])
    """
    # --- already present ---
    if 'spatial' in adata.obsm and isinstance(adata.obsm['spatial'], np.ndarray) \
            and adata.obsm['spatial'].ndim == 2:
        return adata

    # --- from obs columns ---
    if x_col is not None and y_col is not None:
        if x_col not in adata.obs.columns:
            raise ValueError(f"Column '{x_col}' not found in adata.obs.")
        if y_col not in adata.obs.columns:
            raise ValueError(f"Column '{y_col}' not found in adata.obs.")
        adata.obsm['spatial'] = adata.obs[[x_col, y_col]].values.astype(float)
        return adata

    # --- from an external CSV ---
    if csv_path is not None:
        coords_df = pd.read_csv(csv_path, index_col=None if barcode_col else 0)
        if barcode_col is not None:
            if barcode_col not in coords_df.columns:
                raise ValueError(f"Barcode column '{barcode_col}' not found in '{csv_path}'.")
            coords_df = coords_df.set_index(barcode_col)
        if coord_cols is not None:
            if len(coord_cols) != 2:
                raise ValueError("'coord_cols' must contain exactly two column names.")
            coords_df = coords_df[coord_cols]
        else:
            numeric_cols = coords_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError("Could not find two numeric columns in the CSV. "
                                 "Specify them with 'coord_cols'.")
            coords_df = coords_df[numeric_cols[:2]]

        # align to adata barcodes
        missing = set(adata.obs_names) - set(coords_df.index)
        if missing:
            raise ValueError(f"{len(missing)} barcodes in adata are missing from the CSV.")
        adata.obsm['spatial'] = coords_df.loc[adata.obs_names].values.astype(float)
        return adata

    raise ValueError(
        "adata.obsm['spatial'] is not set. Provide either 'x_col'/'y_col', "
        "or 'csv_path' to supply spatial coordinates."
    )


def _Cal_Spatial_Net(adata, k_cutoff=None, max_neigh=30):
    """
    Construct the spatial neighbor networks using KNN.
    Parameters

    adata : anndata.AnnData
        The AnnData object containing the spatial data.
    """

    print('Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='auto').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    indices = indices[:, 1:k_cutoff + 1]
    distances = distances[:, 1:k_cutoff + 1]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
    print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    X = pd.DataFrame(adata.layers['counts'].toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    # add self-loop
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + np.eye(G.shape[0])  # add self-loop
    adata.uns['adj'] = G

def _save_inducing_points(args, inducing_points_dict):
    """
    Write the inducing points to a file.
    """
    # Convert the dictionary into a DataFrame directly
    inducing_points_list = []
    for key, value in inducing_points_dict.items():
        # Create a DataFrame for each key-value pair
        df = pd.DataFrame(value)
        df.columns = ['norm-pixel_x', 'norm-pixel_y']
        df["timepoint"] = key  # Add the key as a new column
        inducing_points_list.append(df)
    # Combine all DataFrames into a single DataFrame
    inducing_points_df = pd.concat(inducing_points_list, ignore_index=True)

    # Save to a file
    inducing_points_df.to_csv(args.output_dir+os.sep+args.prefix+"inducing_points.csv", index=False)


def _sparkx(count, location, genenames, option='mixture', num_cores=1):
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
    sparkx_res = _sparkx_sk(count, final_location, num_cores=num_cores)
    sparkx_res.index = genenames
    sparkx_list.append(sparkx_res)
    if option == 'mixture':
        # --- record run time
        start_time = time.time()
        for iker in range(5):
            print(f"## Testing With Gaussian Kernel {iker+1}")
            final_location = _transloc_func_vec(location, lker=iker, transfunc="gaussian")
            sparkx_list.append(_sparkx_sk(count, final_location, num_cores=num_cores))
        for iker in range(5):
            print(f"## Testing With Cosine Kernel {iker+1}")
            final_location = _transloc_func_vec(location, lker=iker, transfunc="cosine")
            sparkx_list.append(_sparkx_sk(count, final_location, num_cores=num_cores))
        end_time = time.time()
        print(f"Time taken for mixture kernels: {end_time - start_time:.2f} seconds")
    # --- combine p-values from different kernels using ACAT ---
    allstat = np.column_stack([x["stat"] for x in sparkx_list])
    allpvals = np.column_stack([x["pval"] for x in sparkx_list])
    allstat = pd.DataFrame(allstat, index=genenames)    # shape: n_genes × n_methods
    allpvals = pd.DataFrame(allpvals, index=genenames)
    comb_pval = allpvals.apply(_ACAT, axis=1)
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

def _cluster_SVGs(SVG_mat, k=10):
    # --- obtain result from Scanpy
    # record cluster time
    start_time = time.time()
    SVG_adata = sc.AnnData(X=SVG_mat)
    sc.tl.pca(SVG_adata)
    sc.pp.neighbors(SVG_adata, n_neighbors=100, n_pcs=30, method='gauss')
    # sc.pp.neighbors(SVG_adata, n_neighbors=100, n_pcs=30)
    # D = SVG_adata.obsp["distances"].tocoo()
    # W_data = 1.0 / (1.0 + D.data)
    # W = sp.sparse.csr_matrix((W_data, (D.row, D.col)), shape=D.shape)
    # W = W + W.T # make undirected
    # SVG_adata.obsp["connectivities"] = W
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
    end_time = time.time()
    print(f"Time taken for clustering SVGs: {end_time - start_time:.2f} seconds")
    return SVG_adata.obs[f"louvain_{resolution}"].values

def _sparkx_sk(counts, infomat, num_cores=1):
    """Simplified SPARK-X without covariate matrix
    """
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
            _sparkx_pval,
            [(i, vec_ylam, Klam, vec_stat) for i in range(counts.shape[1])]
        )    
    vec_daviesp = np.array(vec_daviesp)  # convert to NumPy array
    res_sparkx = pd.DataFrame({
        'stat': vec_stat,
        'pval': vec_daviesp
    })
    return res_sparkx

def _sparkx_pval(igene,lambda_G,lambda_K,allstat):
    '''
    Calculate p-values for SPARKX

    Parameters
    ----------
    igene: int
        Gene index
    lambda_G : array-like
        Y (lambda)
    lambda_K : array-like
        Eigenvalue of the gene (lambda)
    allstat : array-like
        Statistics for calculating p-values

    Returns
    -------
    float
        P-value P[Q > q]
    '''
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
            pout = _liu(allstat[igene], Zsort)
    except Exception:
        pout = _liu(allstat[igene], Zsort)
    return pout


def _liu(q, lambdas, h=None, delta=None):
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

def _ACAT(pvals, weights=None):
    # --- converted by ChatGPT, original code from: https://github.com/xzhoulab/SPARK/blob/a8b4bf27b804604dfda53da42992f100b8e4e727/R/sparkx.R#L307
    
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

def _transloc_func_vec(coord, lker, transfunc="gaussian"):
    """
    Transform locations
    """
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
