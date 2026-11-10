# Retail demand project status

The current executed pipeline creates lag and rolling demand features from earlier observations, holds out the final three calendar months, fits a bottom-level gradient boosting model, and sums its predictions to store, product, and total levels. See `README.md` for the validation protocol, current metrics, dataset provenance, and limitations.

The earlier project summary described leakage-affected scores, an unvalidated sequence model, and safety-stock recommendations. Those claims have been removed from the current summary. The Git history retains the original document for archival purposes.
