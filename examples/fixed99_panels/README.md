# Frozen feature panels for the three fixed-99 examples

Each dataset directory contains the exact per-timepoint gene lists, their
sorted union, the ranked FDR-significant SPARK-X candidate pool per timepoint,
and `spots.tsv` with the spot order and coordinates used by the validated
input. `panel.json` records counts and source hashes. No tissue labels or
marker identities are used in these assets.

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
quotas, candidate ranks, and Mouse prefix lists. Python then recomputes the
12 KMeans pseudoregions and balanced one-versus-rest gene ranking (or uses
the full Mouse t2 prefix), and requires each computed list to equal its saved
list exactly. It selects their union from raw counts and runs
`normalize_total(target_sum=1e-4)`, `log1p`, then `scale` separately for each
timepoint. No R installation or full SPARK-X CSV table is needed at runtime.
`sparkx_candidates_*.txt` are immutable ranked snapshots from the original
R SPARK-X 1.1.1 run, converted once by `export_candidate_ranks.py` with source
table hashes checked. This is an exact **historical-panel replay in Python**;
it does not imply that de novo Python SPARK-X has identical R p-values.

## Select a new panel entirely in Python

Omit `--feature_panel` to run the bundled Python SPARK-X and pseudoregion
ranking on raw counts:

```sh
SpaDOT preprocess --data new_data.h5ad --output_dir new_output \
  --n_features 500,500,1000
```

The counts map to timepoints in natural label order. Supply exactly one
integer of at least 100 per timepoint. A numerical check substituting Python
SPARK-X into the historical selection script reproduced seven of ten lists;
it did **not** test this complete CLI selector. The CLI candidate-pool and
historical-prefix choices can yield substantially different panels. In the
fixed-99 comparison, fresh Python panels did not preserve Axolotl MP/DP or
Mouse Embryo Heart/Liver separation. Use `--feature_panel` when reproducing
the reported results.
