# SpaDOT

[![PyPI version](https://badge.fury.io/py/SpaDOT.svg?icon=si%3Apython)](https://pypi.org/project/SpaDOT/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) 


Software for the manuscript "Optimal Transport Modeling Uncovers Spatial Domain Dynamics in Spatiotemporal Transcriptomics". First trial in wrapping a code pipeline using GitHub Copilot, inspired by [Robert Langefeld](https://rlangefe.github.io/) from the [Xiang Zhou Lab](https://xiangzhou.github.io/).

SpaDOT is an open-source computational method for identifying spatial domains and inferring their temporal dynamics across time points for spatiotemporal transcriptomics studies. Detailed documentation and tutorials are available at: https://marvinquiet.github.io/SpaDOT/.

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

