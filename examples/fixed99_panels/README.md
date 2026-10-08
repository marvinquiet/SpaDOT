# Frozen feature panels for the three fixed-99 examples

Each dataset directory contains the exact per-timepoint gene lists, their
sorted union, and `spots.tsv` with the spot order and coordinates used by the
validated input. `panel.json` records counts and source hashes. No tissue
labels or marker identities are used in these assets.

| Dataset | Genes selected per timepoint | Union genes |
|---|---|---:|
| Chicken Heart | 500 / 500 / 1,000 / 1,000 | 1,653 |
| Axolotl Telencephalon | 500 / 500 / 1,000 | 1,250 |
| Mouse Embryo | 3,000 / 3,000 / 3,000 | 4,638 |

The original panel development used mitochondrial/ribosomal-filtered
SPARK-X FDR candidates, 12 KMeans pseudoregions formed from eight expression
PCs and spatial coordinates, balanced one-versus-rest contrasts, and
protection of the top 100 SPARK-X genes per timepoint. The lists were frozen
before examining marker overlap. Panel sizes were then chosen after
development-set regional evaluation and need validation on new samples.

## Reproduce the evaluated input without R

The three example notebooks call `SpaDOT preprocess` with `--feature_panel`
and `--n_features`. For example, from the repository root:

```sh
SpaDOT preprocess --data examples/ChickenHeart_data/ChickenHeart.h5ad \
  --output_dir examples/fixed99_runs/ChickenHeart \
  --n_features 500,500,1000,1000 \
  --feature_panel examples/fixed99_panels/ChickenHeart
```

The command checks the raw-data and panel hashes, spot order, timepoint
quotas, and selected-gene union. It selects those genes from raw counts and
runs `normalize_total(target_sum=1e-4)`, `log1p`, then `scale` separately for
each timepoint. The same union is present at every timepoint. No R installation
or SPARK-X table is needed for this exact reproduction path.

## Select a new panel entirely in Python

Omit `--feature_panel` to run the bundled Python SPARK-X and pseudoregion
ranking on raw counts:

```sh
SpaDOT preprocess --data new_data.h5ad --output_dir new_output \
  --n_features 500,500,1000
```

The counts map to timepoints in natural label order. Supply exactly one
integer of at least 100 per timepoint. The Python path does not guarantee
identical gene identities to these frozen evaluation panels: a full comparison
reproduced seven of ten timepoint lists exactly, while small numerical
SPARK-X ranking differences changed the other three. Use `--feature_panel`
when reproducing the reported results.
