# Current deliverables

- `retail_demand_forecasting.py`: executable exploratory analysis and one-step observed-history backtest.
- `model_performance_metrics.csv`: rerun model and observed-history lag-7 naive metrics.
- `forecast_results.csv`: bottom-level held-out predictions from the rerun.
- `tests/test_backtest.py`: focused leakage and bottom-up consistency checks.
- `README.md`: setup, provenance, validation method, metrics, and limits.

The pipeline does not validate a multi-step purchasing forecast or an operational safety-stock policy. The prior deliverables list and its leakage-affected scores remain available only in Git history.
