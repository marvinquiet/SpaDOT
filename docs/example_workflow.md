# Raw-data feature selection for the three SpaDOT examples

The following are the panels used to evaluate the example training configuration.
Counts refer to genes selected **per timepoint**; the input matrix contains
the sorted union across all timepoints. Gene selection excluded annotation
labels and marker identities. The choice among panel rules followed repeated
evaluation against development annotations, so these counts are exploratory
and require validation on new samples.

| Dataset | Technology | Spots by timepoint | Selected genes by timepoint | Union genes |
|---|---|---|---|---:|
| Chicken Heart | 10x Visium | 747 / 1,966 / 1,916 / 1,967 | 500 / 500 / 1,000 / 1,000 | 1,653 |
| Axolotl Telencephalon | Stereo-seq | 1,477 / 2,929 / 4,410 | 500 / 500 / 1,000 | 1,250 |
| Mouse Embryo | 10x Visium | 2,397 / 2,312 / 3,546 | 3,000 / 3,000 / 3,000 | 4,638 |

**There is no validated spot-count-only rule shared by all three.** Chicken
Heart t1 and t2 have similar spot counts but use 500 and 1,000 genes; Mouse
Embryo uses 3,000 genes at each timepoint. A 3,000-spot cutoff describes the
chosen Axolotl panel, but it does not predict the Chicken or Mouse choices.
Use the table as dataset-specific presets, not as a general law.

## Selection and preprocessing

1. Starting with raw counts, remove the previously specified mitochondrial
   and ribosomal genes. Run SPARK-X separately at each timepoint and retain
   FDR-significant candidates.
2. Form 12 unlabeled pseudoregions per timepoint using eight expression PCs
   and spatial coordinates with KMeans. Rank candidate genes by balanced
   one-versus-rest pseudoregion contrasts, protecting the top 100 SPARK-X
   genes at every timepoint.
3. Take the per-timepoint quotas above, freeze their sorted union, and select
   that union from the **raw counts**. For each timepoint independently, run
   `normalize_total(target_sum=1e-4)`, `log1p`, then `scale`. The same union
   genes are present in all timepoints. The source input audits independently
   reconstructed the resulting matrices with maximum absolute difference 0.

`SpaDOT preprocess --n_features 500,500,1000` runs the bundled Python
SPARK-X and pseudoregion selection. The counts must match the number of
observed timepoints and are assigned in natural timepoint-label order.
Each count must be at least 100. From the FDR-significant genes, the Python
selector ordinarily uses a candidate pool of
`min(significant_count, max(1100, quota + min(quota, 1100)))` and the top
`min(1100, candidate_pool)` candidates for PCA. It then protects the top
100 SPARK-X genes and fills the remaining quota by balanced pseudoregion
contrasts. If `--n_features` is omitted, the earlier Python
SCTransform/Louvain selector remains available for compatibility.
The three notebooks instead pass small `--selection_plan` JSON files
containing only a PCA pool and ordered `[quota, candidate_pool]` stages for
each timepoint. The stages preserve genes selected earlier as a prefix.
The corrected Python SPARK-X calculation and these numeric plans regenerated
all ten reported per-timepoint lists exactly from the corresponding raw
H5AD files, without reading archived R ranks or gene lists during
selection. The selected union sizes are shown above. Other inputs or
selection settings may give different genes and spatial results.
When a raw file interleaves timepoints, preprocessing groups its spots by
timepoint while keeping their order within each slice. An already grouped
file retains its original spot order. Training samples inducing points by
row index, so this ordering is part of reproducible model input.

## Training and readout

Use `epoch_start: 1`, `epoch_stop_exclusive: 100`,
`kl_cycle_epochs: 30`, seed 2026, the built-in NumPy OT solver,
and the provided model architecture.
SpaDOT 1.0.5 includes only this Python/NumPy OT backend. It has no
platform-specific `libot.so` dependency.
Use the executed epoch-99 latent for readout. For fixed region counts,
`SpaDOT analyze --n_clusters ... --seed 2026 --n_init 10` uses KMeans++
and Lloyd updates. Dataset-specific K values used in the evaluations were
Chicken Heart 5/7/7/6, Axolotl 6/7/8, and Mouse Embryo 5/5/5; confirm K from
the frozen run manifest before reproducing an ARI.

The historic 99-update experiments retained a 300-update training horizon
only to get a 30-epoch KL period. Version 1.0.5 separates the cycle period
from the actual epoch bounds. In a fresh raw-data check, Mouse Embryo and
Axolotl reproduced the saved 99-epoch loss and latent arrays exactly;
Chicken Heart reproduced its gene panel and expression matrix exactly, with
a small ARI difference because the raw coordinates use another axis
orientation. These checks use one seed and do not establish generalization.
