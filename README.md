# Retail Demand Forecasting

An educational demand-forecasting pipeline on a tracked Kaggle retail dataset. It compares ARIMA on sample products and a bottom-level gradient boosting model, then aggregates store/SKU predictions into coherent store, product, and total daily forecasts. It is not validated for purchasing or inventory decisions.

## Data and provenance

The input is `archive (2)/sales_data.csv` (tracked in this repository), described by the original project as the Kaggle Retail Inventory and Demand Forecasting Dataset. Check its original license and collection method before reuse. The input spans 2022–2024 and includes `Date`, `Store ID`, `Product ID`, and `Demand`. Rows are sorted by store, SKU and date before lag construction. If more than one observation exists per store/SKU/date, define an aggregation policy before interpreting rows as daily forecasts.

## Reproduce

Use a supported Python environment, install `requirements.txt`, then run:

```bash
python retail_demand_forecasting.py
python -m unittest discover -s tests
```

The full script writes plots under `visualizations/`, `model_performance_metrics.csv`, and `forecast_results.csv`. Run from the repository root. Its legacy EDA and ARIMA stages can be slow on the full dataset.
The forecast CSV includes date, store, and product keys so the bottom-level predictions can be regrouped and checked against the code's bottom-up totals.
The base requirements install the portable sklearn fallback. Install optional LightGBM separately only when its native OpenMP runtime is available. The PR smoke workflow checks leakage and aggregation using synthetic data.

## Validation protocol

The gradient boosting model uses only earlier `Demand` values within each store/SKU as lag and rolling features. It uses LightGBM when its native runtime is available and sklearn HistGradientBoosting otherwise; the committed rerun below used sklearn. It excludes same-day sales, inventory and order fields whose availability at forecast time is not established. The last three calendar months form the holdout. MASE is scaled by the mean one-step naive error across training store/SKU series. Bottom-up store/product/total values are sums of the bottom-level holdout predictions.

This is a one-step, observed-history backtest: each holdout row can use earlier *observed* holdout demand. It does not demonstrate a multi-step forecast issued at the split date. The sequence model and legacy safety-stock calculation remain unvalidated demonstrations and are excluded from the main pipeline.

On the tracked input, the 2026-11-10 rerun with the sklearn fallback achieved MAE 30.676, RMSE 39.330, MASE 0.746, and sMAPE 31.457% on 9,200 holdout rows. A lag-7 observed-history comparator on those same rows had MAE 43.446. These are one-step results, not forecasts issued 90 days in advance. The previous 0.247 MASE and 12.805% sMAPE are invalid historical figures. Multiple rolling-origin splits and a true multi-step seasonal-naive comparison remain necessary before operational claims.

## License

No software license is included. Dataset redistribution rights must be checked with the original publisher.

**Repository description recommendation:** “Retail demand backtesting with leakage-free observed-history lags and bottom-up forecast aggregation.”
