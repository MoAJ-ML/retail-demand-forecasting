# Retail Demand Forecasting System

A comprehensive multi-store, multi-SKU demand forecasting system with hierarchical reconciliation, implementing multiple forecasting models and safety stock calculations for inventory planning.

## 🎯 Project Overview

This project builds a complete retail demand forecasting pipeline that:
- Analyzes retail sales data across multiple stores and product categories
- Implements multiple forecasting models (ARIMA, LightGBM, Sequence-based)
- Applies hierarchical reconciliation for consistent forecasts
- Calculates safety stock recommendations for inventory planning
- Provides comprehensive evaluation using MASE and sMAPE metrics

## 📊 Dataset Information

- **Source**: Kaggle Retail Inventory and Demand Forecasting Dataset
- **Size**: 76,000 records
- **Time Period**: January 2022 - January 2024
- **Features**: 16 columns including Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic, and Demand
- **Stores**: Multiple stores across different regions
- **Products**: Various product categories (Electronics, Clothing, Groceries, Toys, Furniture)

## 🚀 Key Features

### 1. Exploratory Data Analysis (EDA)
- Comprehensive data overview and statistics
- Demand distribution analysis across stores and categories
- Seasonality and trend analysis
- Correlation analysis between features
- Missing data analysis

### 2. Forecasting Models
- **ARIMA/SARIMAX**: Time series forecasting for individual SKUs
- **LightGBM**: Gradient boosting with lag and rolling features
- **Sequence Model**: Linear regression with sequence-based features (simplified LSTM alternative)

### 3. Hierarchical Forecasting
- Bottom-up reconciliation ensuring forecasts add up correctly across hierarchy levels
- Store-SKU level aggregation
- Consistent forecasting across different aggregation levels

### 4. Model Evaluation
- **MASE (Mean Absolute Scaled Error)**: Scale-independent accuracy measure
- **sMAPE (Symmetric Mean Absolute Percentage Error)**: Percentage-based error metric
- **RMSE & MAE**: Standard regression metrics
- Comprehensive model comparison and visualization

### 5. Safety Stock Calculator
- Implements safety stock formula: `Safety Stock = Z × σ(L)`
- Service level-based calculations (95% default)
- Store-SKU specific recommendations
- Risk-based inventory planning

## 📈 Model Performance Results

| Model | MASE | sMAPE (%) | RMSE | MAE |
|-------|------|-----------|------|-----|
| LightGBM | 0.247 | 12.805 | 15.368 | 11.966 |
| Sequence Model | 0.635 | 32.104 | 38.591 | 30.626 |
| ARIMA (Sample SKUs) | 1.027-1.125 | 20.894-31.645 | - | - |

**Best Performing Model**: LightGBM with comprehensive feature engineering

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
# Clone or download the project
# Install required packages
pip install -r requirements.txt

# Run the complete pipeline
python retail_demand_forecasting.py
```

### Required Packages
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.1.0
lightgbm>=3.3.0
xgboost>=1.6.0
statsmodels>=0.13.0
scipy>=1.9.0
joblib>=1.2.0
```

## 📁 Project Structure

```
Retail2/
├── retail_demand_forecasting.py    # Main forecasting pipeline
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── archive (2)/
│   └── sales_data.csv              # Input dataset
├── visualizations/                 # Generated plots and charts
│   ├── eda_overview.png
│   ├── correlation_matrix.png
│   ├── seasonality_analysis.png
│   ├── missing_data_analysis.png
│   ├── model_comparison.png
│   ├── model_performance.png
│   ├── forecast_analysis.png
│   └── safety_stock_analysis.png
├── forecast_results.csv            # Model predictions and residuals
├── model_performance_metrics.csv   # Evaluation metrics
└── safety_stock_recommendations.csv # Safety stock calculations
```

## 🔧 Usage

### Running the Complete Pipeline
```python
from retail_demand_forecasting import RetailDemandForecaster

# Initialize the forecaster
forecaster = RetailDemandForecaster('archive (2)/sales_data.csv')

# Run the complete pipeline
forecaster.run_complete_pipeline()
```

### Individual Components
```python
# Load and explore data
forecaster.load_data()
forecaster.exploratory_data_analysis()

# Preprocess data
forecaster.preprocess_data()

# Run individual models
forecaster.arima_forecasting()
forecaster.lightgbm_forecasting()
forecaster.lstm_forecasting()

# Evaluate and visualize
forecaster.evaluate_models()
forecaster.safety_stock_calculator()
```

## 📊 Key Insights

### Data Characteristics
- **No Missing Data**: Clean dataset with complete records
- **Outlier Handling**: 986 outliers removed using IQR method
- **Temporal Split**: Last 3 months (Oct 2023 - Jan 2024) used for testing
- **Hierarchical Structure**: Store-SKU combinations with proper aggregation

### Model Performance
- **LightGBM** outperforms other models with MASE of 0.247
- **Feature Engineering** with lag and rolling features significantly improves performance
- **ARIMA** shows good performance for individual SKUs but requires sufficient data
- **Sequence Model** provides reasonable baseline performance

### Safety Stock Insights
- Top products requiring highest safety stock: P0018, P0007, P0010
- Store S005 shows highest safety stock requirements
- Service level of 95% provides balanced inventory protection

## 🎯 Business Applications

### Inventory Management
- **Demand Forecasting**: Accurate predictions for procurement planning
- **Safety Stock Optimization**: Risk-based inventory levels
- **Store-Specific Planning**: Tailored recommendations per location

### Strategic Planning
- **Category Performance**: Electronics and Clothing show highest demand
- **Seasonal Patterns**: Clear seasonal trends identified
- **Promotional Impact**: Promotion effects quantified

### Operational Efficiency
- **Automated Forecasting**: End-to-end pipeline for regular updates
- **Multi-Model Approach**: Robust predictions through ensemble methods
- **Hierarchical Consistency**: Coherent forecasts across all levels

## 🔮 Future Enhancements

1. **Deep Learning Models**: Implement proper LSTM/GRU with TensorFlow compatibility
2. **Advanced Reconciliation**: Integrate scikit-hts for sophisticated hierarchical methods
3. **External Factors**: Incorporate weather, economic indicators, and competitor data
4. **Real-time Updates**: Streaming data integration for live forecasting
5. **A/B Testing**: Framework for model comparison and selection
6. **Automated Hyperparameter Tuning**: Optimize model parameters automatically

## 📝 Technical Notes

### Model Selection Rationale
- **ARIMA**: Traditional time series approach, good for individual SKUs
- **LightGBM**: Handles complex feature interactions, robust to outliers
- **Sequence Model**: Simplified deep learning approach, good baseline

### Evaluation Metrics
- **MASE**: Scale-independent, allows comparison across different SKUs
- **sMAPE**: Percentage-based, intuitive for business stakeholders
- **RMSE/MAE**: Standard regression metrics for model comparison

### Data Preprocessing
- **Outlier Removal**: IQR-based method preserves data integrity
- **Lag Features**: 1, 2, 3, 7, 14-day lags capture temporal patterns
- **Rolling Features**: 7, 14, 30-day windows for trend analysis

## 👥 Author

**Data Science Expert** - Comprehensive retail forecasting system implementation

## 📄 License

This project is for educational and research purposes. Please ensure proper attribution when using or modifying the code.

---

*Generated as part of a comprehensive retail demand forecasting project demonstrating end-to-end data science pipeline implementation.*
