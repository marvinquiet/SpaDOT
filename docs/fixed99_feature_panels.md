# Feature panels for the fixed 99-epoch SpaDOT recipe

The following are the frozen panels used to evaluate the training recipe.
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
selector uses a candidate pool of
`min(significant_count, max(1100, quota + min(quota, 1100)))` and the top
`min(1100, candidate_pool)` candidates for PCA. It then protects the top
100 SPARK-X genes and fills the remaining quota by balanced pseudoregion
contrasts. If `--n_features` is omitted, the earlier Python
SCTransform/Louvain selector remains available for compatibility.
For **exact reproduction** of a reported input, also pass its
`--feature_panel examples/fixed99_panels/<dataset>` directory. The bundled
panel stores the historical R SPARK-X ranked candidate pools as immutable
text snapshots, not as executable R code. Python checks their hashes,
recomputes the KMeans pseudoregion contrasts and per-timepoint lists, and
requires an exact match with the saved genes before preprocessing. The
original R tables are needed only by the one-time Python exporter that made
these snapshots. No R installation or R process is needed at runtime. The
three notebooks use this exact mode.

The Python SPARK-X p-values were close to the original R values in an isolated
numerical check. Substituting only those p-values into the historical
selection script regenerated seven of ten per-timepoint lists exactly; the
other three shared 446/500 Chicken Heart t1, 495/500 Axolotl t0, and
2,871/3,000 Mouse Embryo t0 genes. This check did **not** test the complete
`SpaDOT preprocess --n_features` path, which also differs in candidate-pool
and historical-prefix choices. Its fresh panels had substantially lower
overlap with the frozen lists. In matched 99-epoch seed-2026 trials, global
mean ARI was .521/.537/.559 for Chicken Heart/Axolotl/Mouse Embryo versus
.526/.611/.594 for the frozen panels. The fresh Axolotl panel merged MP and
DP at t2; the fresh Mouse panel merged Heart and Liver at t0 and t1. These
development results do not establish generalization. A fresh Python
selection is a new input panel; use `--feature_panel` to reproduce evaluated
results.

## Training and readout

Use `epoch_start: 1`, `epoch_stop_exclusive: 100`,
`kl_cycle_epochs: 30`, seed 2026, the built-in NumPy OT solver,
and the provided model architecture.
The release candidate includes only this Python/NumPy OT backend. It has no
platform-specific `libot.so` dependency.
Use the executed epoch-99 latent for readout. For fixed region counts,
`SpaDOT analyze --n_clusters ... --seed 2026 --n_init 10` uses KMeans++
and Lloyd updates. Dataset-specific K values used in the evaluations were
Chicken Heart 5/7/7/6, Axolotl 6/7/8, and Mouse Embryo 5/5/5; confirm K from
the frozen run manifest before reproducing an ARI.

The historic 99-update experiments retained a 300-update training horizon
only to get a 30-epoch KL period. This release separates the cycle period
from the actual epoch bounds.
The new package should be benchmarked against the three frozen inputs before
claiming numerical equivalence, including the spatial Heart/Liver and MP/DP
checks. Matching the KL schedule alone does not prove an identical run.
