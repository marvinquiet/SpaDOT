import anndata as ad
import numpy as np
import pandas as pd

from SpaDOT.utils._analyze_utils import KMeans_Clustering


def test_kmeans_accepts_categorical_timepoints():
    data = ad.AnnData(
        X=np.array([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0],
                    [0.0, 0.1], [0.1, 0.1], [4.0, 4.1], [4.1, 4.1]]),
        obs=pd.DataFrame(
            {'timepoint': pd.Categorical(['t1'] * 4 + ['t0'] * 4)},
            index=[str(i) for i in range(8)],
        ),
    )
    result = KMeans_Clustering(data, [2, 2], seed=2026, n_init=3)
    assert result.n_obs == 8
    assert result.obs['timepoint'].astype(str).tolist() == ['t0'] * 4 + ['t1'] * 4
    assert result.obs.groupby('timepoint', observed=True)['kmeans'].nunique().to_dict() == {
        't0': 2, 't1': 2,
    }
