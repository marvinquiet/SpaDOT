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

The upstream selection used mt/ribosomal-filtered R SPARK-X FDR candidates,
KMeans-12 pseudoregions formed from expression PCs and spatial coordinates,
balanced one-versus-rest gene contrasts, and protection of the top 100
SPARK-X genes per timepoint. The lists were frozen before examining marker
overlap. Choosing these panel sizes followed development-set regional
results, so this is a reproducibility example rather than evidence of
out-of-sample generalization.

`fixed99_example.prepare_panel` applies the lists to the raw example counts,
restores the audited spot order and coordinates, and runs
`normalize_total(target_sum=1e-4)`, `log1p`, then `scale` separately by
timepoint. The generic `SpaDOT.preprocess` feature-selection option uses a
different rule and does not recreate these panels.

## Rebuild a panel

Run this from the repository root with the raw example data available. This
Chicken Heart command shows the three stages; replace the dataset paths for
Axolotl or Mouse Embryo. The output directories must be new. R requires
SPARK 1.1.1 and Matrix.

```sh
python examples/select_pseudoregion_svgs.py \
  --raw examples/ChickenHeart_data/ChickenHeart.h5ad \
  --panel-dir examples/fixed99_panels/ChickenHeart \
  --output-dir /tmp/chicken_sparkx_input --export-r-inputs
Rscript examples/run_filtered_sparkx.R \
  /tmp/chicken_sparkx_input /tmp/chicken_sparkx_tables
python examples/select_pseudoregion_svgs.py \
  --raw examples/ChickenHeart_data/ChickenHeart.h5ad \
  --panel-dir examples/fixed99_panels/ChickenHeart \
  --sparkx-dir /tmp/chicken_sparkx_tables \
  --output-dir /tmp/chicken_rebuilt_panel
```

The selector checks the raw-data and SPARK-X table hashes, then requires
every regenerated gene list to match the frozen panel exactly. Archived
SPARK-X tables with the recorded hashes can also be supplied for an exact
selection replay. The example notebooks use the frozen lists so their input
can be reproduced without rerunning R.
