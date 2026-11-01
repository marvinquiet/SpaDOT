# SpaDOT 1.0.6

- Final spot-level OT uses epsilon=0.01, lambda1=0.1, lambda2=5, and three growth updates.
- Convergence diagnostics operate on log scalings to avoid overflow; non-finite objectives and iteration-limit failures raise errors instead of returning unverified maps.
- Final analysis uses the stabilized SpaDOT NumPy solver within WOT.
- The three example notebooks contain refreshed analysis and figures from their saved embeddings. Their existing SpaDOT 1.0.5 preprocessing/training outputs are retained and explicitly identified.
- Transition dotplots retain the 0.20 display threshold. Model architecture, feature selection, and training configuration are unchanged.

The numerical solver is also used by training. Small deterministic OT regression fixtures pass, but this release refresh did not rerun end-to-end model training.
