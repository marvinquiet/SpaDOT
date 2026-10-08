# SpaDOT

[![PyPI version](https://badge.fury.io/py/SpaDOT.svg?icon=si%3Apython)](https://pypi.org/project/SpaDOT/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) 


Software for the manuscript "Optimal Transport Modeling Uncovers Spatial Domain Dynamics in Spatiotemporal Transcriptomics". First trial in wrapping a code pipeline using GitHub Copilot, inspired by [Robert Langefeld](https://rlangefe.github.io/) from the [Xiang Zhou Lab](https://xiangzhou.github.io/).

SpaDOT is an open-source computational method for identifying spatial domains and inferring their temporal dynamics across time points for spatiotemporal transcriptomics studies. Detailed documentation and tutorials are available at: https://marvinquiet.github.io/SpaDOT/.

### Fixed 99-epoch training recipe

The bundled `SpaDOT/config.yaml` starts at epoch 1 and executes while
`epoch < 100`, for exactly 99 optimizer updates (epochs 1 through 99).
The SVGP KL coefficient follows a 30-epoch linear cycle, controlled by
`kl_cycle_epochs: 30`; its period is independent of the training length.
The packaged trainer does not use reconstruction-loss patience or restore an
earlier checkpoint. The saved model and `latent.h5ad` are from executed epoch
99. Set `epoch_start` and `epoch_stop_exclusive` in custom configs; their
difference is the number of optimizer updates. `maxiter` is no longer a
training setting. A custom config without `kl_cycle_epochs` retains the
historical ten cycles over its training length.

The evaluated workflow used seed 2026, NumPy OT, and final KMeans with
`random_state=2026`, `n_init=10`, `init='k-means++'`, and `algorithm='lloyd'`.
The bundled OT solver is Python/NumPy only. It does not load or ship
`libot.so`, and no backend flag is needed. Older configurations with
`use_Py: true` and `use_C: false` still work; requesting the removed
C backend raises an explicit error.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=2026`, and numerical
thread limits to 1 before launching GPU training for reproducibility.
The evaluated input panels were:

| Dataset | Spots by timepoint | Selected genes by timepoint | Unique genes in input |
|---|---|---|---:|
| Chicken Heart | 747 / 1,966 / 1,916 / 1,967 | 500 / 500 / 1,000 / 1,000 | 1,653 |
| Axolotl Telencephalon | 1,477 / 2,929 / 4,410 | 500 / 500 / 1,000 | 1,250 |
| Mouse Embryo | 2,397 / 2,312 / 3,546 | 3,000 / 3,000 / 3,000 | 4,638 |

The counts above are dataset-specific exploratory presets. The spot count
alone did not determine gene quota across these datasets. See
`docs/fixed99_feature_panels.md` in the source repository for the selection
and preprocessing recipe. `SpaDOT preprocess --n_features 500,500,1000`
uses the bundled Python SPARK-X and pseudoregion selection, with one quota
per timepoint. Add `--feature_panel` for exact reproduction from the frozen
lists. The executable notebooks are
[`ChickenHeart.ipynb`](examples/ChickenHeart.ipynb),
[`AxolotlTelencephalon.ipynb`](examples/AxolotlTelencephalon.ipynb), and
[`MouseEmbryo.ipynb`](examples/MouseEmbryo.ipynb). Each uses the audited
frozen lists in `examples/fixed99_panels/`. The R-free selection procedure is
integrated into `SpaDOT/utils/_preprocess_utils.py`. Fresh Python rankings can
differ slightly from the saved evaluated lists; see the panel documentation.

**Table of Contents**
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [License](#license)
- [Citing Our Work](#citing-our-work)

## System Requirements

### Hardware requirements

It is recommended to run SpaDOT on a GPU to achieve faster execution times.

#### Dependencies

SpaDOT requires the following:

- [python](https://www.python.org/) (version 3.9 recommended)
- [torch](https://pytorch.org/) (v2.5.0)
- [anndata](https://anndata.readthedocs.io/en/latest/) (v0.9.1)
- [scanpy](https://scanpy.readthedocs.io/en/stable/) (v1.9.8)
- [numpy](https://numpy.org/) 
- [pandas](https://pandas.pydata.org/)
- [wot](https://github.com/broadinstitute/wot)
- `cuda toolkit` and `nvidia cudnn` when using GPU resources, more information can be found [here](https://towardsdatascience.com/setting-up-tensorflow-gpu-with-cuda-and-anaconda-onwindows-2ee9c39b5c44)
- [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/), this package is dependent on torch version and cuda version.


**Note:** If using GPU resources, you might also need to install the CUDA Toolkit and NVIDIA cuDNN libraries. Detailed setup instructions can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Please make sure to install a compatible version for PyTorch.

To fully leverage SpaDOT’s capabilities, installing [R](https://www.r-project.org/) and the [SPARK-X](https://xzhoulab.github.io/SPARK/) package is recommended for selecting spatially variable genes.

## Installation

The most convenient way is to install with `pip`.

```shell
pip install SpaDOT
```

To upgrade to a newer release use the `--upgrade` flag.

```shell
pip install --upgrade SpaDOT
```

For more information, a detailed documentation on installation is provided [here](https://marvinquiet.github.io/SpaDOT/).


## License

This project is covered under the **MIT license**.


## Citing Our Work

<!-- For usage of the package and associated manuscript, please cite: 
```BibTex
@article{ma23cellcano,
  title   = {Cellcano: supervised cell type identification for single cell ATAC-seq data},
  author  = {Ma, Wenjing and Lu, Jiaying and Wu, Hao},
  journal = {Nature Communications},
  year    = {2023},
  month   = {Apr.},
  day     = {03},
  volume={14},
  number={1},
  pages={1864},
  issn={2041-1723},
  doi={10.1038/s41467-023-37439-3},
  url={https://doi.org/10.1038/s41467-023-37439-3}
}
``` -->
