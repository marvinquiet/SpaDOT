## SpaDOT Tutorial

**Authors:**  
[Wenjing Ma](https://marvinquiet.github.io/) (wenjinma@umich.edu), Department of Biostatistics, University of Michigan  
Siyu Hou, Department of Biostatistics, University of Michigan  
[Lulu Shang](https://lulushang.org/), Department of Biostatistics, MD Anderson Cancer Center  
[Jiaying Lu](https://lujiaying.github.io/), Center for Data Science, School of Nursing, Emory University  
[Xiang Zhou](https://xiangzhou.github.io/), Department of Biostatistics, University of Michigan  

**Maintainer:** [Wenjing Ma](https://marvinquiet.github.io/) (wenjinma@umich.edu)

**Latest revision:** 04/28/2025

### Introduction

Spatiotemporal transcriptomics is an emerging and powerful approach that adds a temporal dimension to traditional spatial transcriptomics, thus enabling the characterization of dynamic changes in tissue architecture during development or disease progression. Tissue architecture is generally organized into spatial domains -- regions within a tissue characterized by relatively similar gene expression profiles that often correspond to distinct biological functions. Crucially, these spatial domains are not static; rather, they undergo complex temporal dynamics during development, differentiation, and disease progression, resulting in emergence, disappearance, splitting, and merging of domains over time. Therefore, we develop SpaDOT (**Spa**tial **DO**main **T**ransition detection), a novel and scalable machine learning method for identifying spatial domains and inferring their temporal dynamics in spatiotemporal transcriptomics studies.

![img](workflow.png)

(The figure illustrates how SpaDOT works. SpaDOT adopts an integration of two complementary encoders, a Gaussian Process kernel and a Graph Attention Transformer, within one variational autoencoder framework to obtain spatially aware latent representations. The latent representations are further constrained by clustering within each time point and optimal transport (OT) coupling across time points, enabling SpaDOT to identify spatial domains and capture domain transition dynamics. )

In this tutorial, we provide detailed instructions for SpaDOT by utilizing two real data applications: a developing chicken heart sequenced by 10X Vision and a developing mouse brain sequenced by Stereo-seq. 

### Installation

**Step 1**: SpaDOT is developed as a Python package. You will have to install Python, and the recommended version is **Python 3.9**. SpaDOT also incorporates an R package [SPARK-X](https://github.com/xzhoulab/SPARK) to perform feature selection as an option. Having the spatial variable genes selected is a practice that we test can generate better results. Therefore, installiation of R and SPARK-X is recommended.

**Step 2**: Use the following command to install SpaDOT:

```shell
pip install SpaDOT
```
The installation will take seconds to finish and the software dependencies have been taken care of by pip. We have tested our package on Windows, MacOS, and Linux. 

**Step 3**: If you have successfully installed SpaDOT, you can try the following command:

```shell
## Check the help documentation of SpaDOT
SpaDOT -h 
```

You should see the following console output:

```
usage: SpaDOT [-h] {preprocess,train,predict} ...

SpaDOT: Spatial DOmain Transition detection for spatiotemporal transcriptomics studies.

positional arguments:
  {preprocess,train,predict}
                        sub-command help.
    preprocess          Perform data preprocessing and feature selection (optional).
    train               Train SpaDOT model and obtain latent space.
    predict             Use obtained latent space to perform domain detection and domain dynamics detection.

optional arguments:
  -h, --help            show this help message and exit
```

**Step 4 (Optional):** If you would like to use SpaDOT with-in program, you can do:

```
import SpaDOT

# load your own data into anndata, with `timepoint` indicating from which time point the data is collected. 
preprocessed_adata = SpaDOT.preprocess(adata)

```





---







### Example 1: developing chicken heart

The developing chicken heart is measured by 10X Visium and collected from four stages: Day 4, Day 7, Day 10 and Day 14. 

**Step 1: obtain example data**

We first downloaded the data from [GSE149457](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149457) and selected `GSM4502482_chicken_heart_spatial_RNAseq_D4_filtered_feature_bc_matrix.h5`, `GSM4502483_chicken_heart_spatial_RNAseq_D7_filtered_feature_bc_matrix.h5`, `GSM4502484_chicken_heart_spatial_RNAseq_D10_filtered_feature_bc_matrix.h5` and `GSM4502485_chicken_heart_spatial_RNAseq_D14_filtered_feature_bc_matrix.h5`. For your convenience, I used the script `process_ChickenHeart.py` provided [here]() to preprocess the data by integrating them into one anndata with `timepoint` in anndata observations (obs) as one-hot encoder indicating four time points, `0`, `1`, `2` and `3` indicate Day 4, Day 7, Day 10 and Day 14, respectively. I have also put the spatial coordinates with keyword `spatial` as a numpy array inside anndata observation metadata (obsm).

**Step 2: perform data preprocessing**



**Step 3: train SpaDOT to obtain latent representations**



**Step 4: infer spatial domains and domain dynamics based on region number**


**Step 5: infer spatial domains and domain dynamics based on Elbow method (Optional)**



### Conclusion

SpaDOT provides efficient and accurate spatial domain detection for spatiotemporal transcriptomics studies and offers insights into domain transition dynamics. Its contributions are three-fold:

1. **Capturing domain dynamics via optimal transport (OT) constraints**: Across time points, OT constraints guide the alignment of functionally similar domains while separating dissimilar ones, enabling SpaDOT to accurately identify both shared and time-specific domains and infer their biological relationships over time.

2. **Modeling both global and local spatial patterns for enhanced structural embedding**: Within each time point, SpaDOT integrates a Gaussian Process (GP) prior and Graph Attention Transformer (GAT) within a variational autoencoder (VAE) framework, capturing both global spatial continuity and local structural heterogeneity.

3. **Eliminating the need to predefine the number of spatial domains**: Unlike existing methods, SpaDOT does not require the number of domains to be specified in advance, reducing dependence on prior knowledge and improving ease of use.

We hope that SpaDOT will be a useful tool for your research.

For questions or comments, please open an issue on [Github](https://github.com/marvinquiet/SpaDOT/issues).

<!--**Citation**-->

