#!/usr/bin/env python3
"""
Retail Demand Forecasting System
================================

A comprehensive multi-store, multi-SKU demand forecasting system with hierarchical reconciliation.
Implements ARIMA, LightGBM, and LSTM models with safety stock calculations.

Author: Data Science Expert
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Deep learning (optional - commented out due to compatibility issues)
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# Hierarchical forecasting
from scipy.optimize import minimize

import os
from datetime import datetime, timedelta
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
# tf.random.set_seed(42)  # Commented out due to TensorFlow compatibility


def create_lag_features(df, lags=(1, 2, 3, 7, 14), windows=(7, 14, 30)):
    """Create features using only earlier observations for each store and SKU."""
    result = df.sort_values(['Store ID', 'Product ID', 'Date']).copy()
    groups = result.groupby(['Store ID', 'Product ID'], sort=False)
    for lag in lags:
        result[f'demand_lag_{lag}'] = groups['Demand'].shift(lag)
    for window in windows:
        result[f'demand_rolling_mean_{window}'] = groups['Demand'].transform(
            lambda series: series.shift(1).rolling(window).mean())
        result[f'demand_rolling_std_{window}'] = groups['Demand'].transform(
            lambda series: series.shift(1).rolling(window).std())
    return result

class RetailDemandForecaster:
    """
    Comprehensive retail demand forecasting system with hierarchical reconciliation.
    """
    
    def __init__(self, data_path):
        """Initialize the forecaster with data path."""
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.safety_stocks = {}
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
    def load_data(self):
        """Load and display basic information about the dataset."""
        print("🔄 Loading retail dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n📊 Dataset Information:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Number of stores: {self.df['Store ID'].nunique()}")
        print(f"Number of products: {self.df['Product ID'].nunique()}")
        print(f"Number of categories: {self.df['Category'].nunique()}")
        
        # Display first few rows
        print(f"\n📋 First 5 rows:")
        print(self.df.head())
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(self.df.describe())
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis."""
        print("\n🔍 Performing Exploratory Data Analysis...")
        
        # Convert Date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(['Date', 'Store ID', 'Product ID'])
        
        # 1. Demand distribution analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.df['Demand'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Demand Distribution')
        plt.xlabel('Demand')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 3, 2)
        self.df['Demand'].plot(kind='box')
        plt.title('Demand Box Plot')
        plt.ylabel('Demand')
        
        plt.subplot(2, 3, 3)
        demand_by_category = self.df.groupby('Category')['Demand'].sum().sort_values(ascending=False)
        demand_by_category.plot(kind='bar', color='lightcoral')
        plt.title('Total Demand by Category')
        plt.xlabel('Category')
        plt.ylabel('Total Demand')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        demand_by_store = self.df.groupby('Store ID')['Demand'].sum().sort_values(ascending=False)
        demand_by_store.plot(kind='bar', color='lightgreen')
        plt.title('Total Demand by Store')
        plt.xlabel('Store ID')
        plt.ylabel('Total Demand')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        daily_demand = self.df.groupby('Date')['Demand'].sum()
        daily_demand.plot(color='purple')
        plt.title('Daily Total Demand Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        
        plt.subplot(2, 3, 6)
        monthly_demand = self.df.groupby(self.df['Date'].dt.to_period('M'))['Demand'].sum()
        monthly_demand.plot(kind='bar', color='orange')
        plt.title('Monthly Total Demand')
        plt.xlabel('Month')
        plt.ylabel('Total Demand')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Seasonality analysis
        self._analyze_seasonality()
        
        # 3. Correlation analysis
        self._analyze_correlations()
        
        # 4. Missing data analysis
        self._analyze_missing_data()
        
        print("✅ EDA completed and visualizations saved!")
    
    def _analyze_seasonality(self):
        """Analyze seasonality patterns."""
        print("\n📅 Analyzing Seasonality...")
        
        # Sample a few SKUs for seasonality analysis
        sample_skus = self.df['Product ID'].unique()[:3]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, sku in enumerate(sample_skus):
            sku_data = self.df[self.df['Product ID'] == sku].groupby('Date')['Demand'].sum()
            
            if len(sku_data) > 30:  # Need sufficient data for decomposition
                decomposition = seasonal_decompose(sku_data, model='additive', period=7)
                
                row, col = i // 2, i % 2
                if row < 2 and col < 2:
                    decomposition.trend.plot(ax=axes[row, col], title=f'Trend - {sku}')
                    axes[row, col].set_ylabel('Demand')
        
        plt.tight_layout()
        plt.savefig('visualizations/seasonality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_correlations(self):
        """Analyze correlations between features."""
        print("\n🔗 Analyzing Correlations...")
        
        # Select numeric columns for correlation
        numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 
                       'Discount', 'Promotion', 'Competitor Pricing', 'Demand']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_missing_data(self):
        """Analyze missing data patterns."""
        print("\n❓ Analyzing Missing Data...")
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percentage
        })
        
        # Check if there are any missing values
        missing_features = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_features) > 0:
            plt.figure(figsize=(10, 6))
            missing_features.plot(kind='bar', y='Missing Percentage')
            plt.title('Missing Data Analysis')
            plt.xlabel('Features')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Missing data summary:\n{missing_features}")
        else:
            print("✅ No missing data found in the dataset!")
            # Create a simple plot showing no missing data
            plt.figure(figsize=(10, 6))
            plt.bar(['No Missing Data'], [0], color='green', alpha=0.7)
            plt.title('Missing Data Analysis - No Missing Values Found')
            plt.ylabel('Missing Percentage (%)')
            plt.ylim(0, 5)
            plt.tight_layout()
            plt.savefig('visualizations/missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for modeling."""
        print("\n🔧 Preprocessing Data...")
        
        # Handle missing values
        print("Handling missing values...")
        # Future-facing backfill and target-based outlier removal would alter
        # the holdout distribution. Preserve the observed rows for evaluation.
        if self.df['Demand'].isna().any():
            raise ValueError('Demand contains missing values; define a training-only imputation policy')
        
        print('Preserving all observed demand values, including peaks')
        
        # Create hierarchical aggregations
        print("Creating hierarchical aggregations...")
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(['Date', 'Store ID', 'Product ID']).set_index('Date')
        
        # Store-SKU level (bottom level)
        store_sku_demand = self.df.groupby(['Store ID', 'Product ID'])['Demand'].resample('D').sum()
        
        # Store level
        store_demand = self.df.groupby('Store ID')['Demand'].resample('D').sum()
        
        # SKU level
        sku_demand = self.df.groupby('Product ID')['Demand'].resample('D').sum()
        
        # Total level
        total_demand = self.df['Demand'].resample('D').sum()
        
        # Split data (last 3 months for testing)
        split_date = self.df.index.max() - pd.DateOffset(months=3)
        
        self.train_data = self.df[self.df.index <= split_date]
        self.test_data = self.df[self.df.index > split_date]
        
        print(f"Training data: {len(self.train_data)} records")
        print(f"Test data: {len(self.test_data)} records")
        print(f"Split date: {split_date}")
        
        # Store hierarchical data
        self.hierarchical_data = {
            'store_sku': store_sku_demand,
            'store': store_demand,
            'sku': sku_demand,
            'total': total_demand
        }
        
        print("✅ Data preprocessing completed!")
    
    def calculate_mase(self, actual, forecast, training_series):
        """Scale holdout MAE by one-step naive error on training observations."""
        train = np.asarray(training_series, dtype=float)
        if len(train) < 2:
            return np.nan
        scale = np.mean(np.abs(np.diff(train)))
        return np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))) / scale if scale else np.nan
    
    def calculate_smape(self, actual, forecast):
        """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
        denominator = np.abs(actual) + np.abs(forecast)
        return 100 * np.mean(np.divide(2 * np.abs(actual - forecast), denominator,
                                       out=np.zeros_like(np.asarray(actual, dtype=float)), where=denominator != 0))
    
    def arima_forecasting(self):
        """Implement ARIMA/SARIMAX forecasting."""
        print("\n📈 Implementing ARIMA Forecasting...")
        
        # Sample a few SKUs for demonstration
        sample_skus = self.df['Product ID'].unique()[:5]
        arima_forecasts = {}
        
        for sku in sample_skus:
            try:
                # Get time series data for this SKU
                sku_data = self.df[self.df['Product ID'] == sku].groupby('Date')['Demand'].sum()
                
                if len(sku_data) < 30:  # Need sufficient data
                    continue
                
                # Split into train/test
                train_size = int(len(sku_data) * 0.8)
                train_ts = sku_data[:train_size]
                test_ts = sku_data[train_size:]
                
                # Fit ARIMA model
                model = ARIMA(train_ts, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Make forecasts
                forecast = fitted_model.forecast(steps=len(test_ts))
                
                # Calculate metrics
                mase = self.calculate_mase(test_ts.values, forecast, train_ts.values)
                smape = self.calculate_smape(test_ts.values, forecast)
                
                arima_forecasts[sku] = {
                    'forecast': forecast,
                    'actual': test_ts.values,
                    'mase': mase,
                    'smape': smape
                }
                
                print(f"ARIMA for {sku}: MASE={mase:.3f}, sMAPE={smape:.3f}")
                
            except Exception as e:
                print(f"Error fitting ARIMA for {sku}: {e}")
                continue
        
        self.models['arima'] = arima_forecasts
        print("✅ ARIMA forecasting completed!")
    
    def lightgbm_forecasting(self):
        """Implement LightGBM forecasting with lag features."""
        print("\n🌳 Implementing LightGBM Forecasting...")
        
        # Same-day sales and inventory fields are not known at forecast time.
        feature_cols = []
        # Apply lag features
        df_with_lags = create_lag_features(self.df.reset_index())
        
        # Encode categorical variables
        le_store = LabelEncoder()
        le_product = LabelEncoder()
        le_category = LabelEncoder()
        
        df_with_lags['Store ID_encoded'] = le_store.fit_transform(df_with_lags['Store ID'])
        df_with_lags['Product ID_encoded'] = le_product.fit_transform(df_with_lags['Product ID'])
        df_with_lags['Category_encoded'] = le_category.fit_transform(df_with_lags['Category'])
        
        # Prepare features
        all_features = feature_cols + [f'demand_lag_{lag}' for lag in [1, 2, 3, 7, 14]] + \
                      [f'demand_rolling_mean_{window}' for window in [7, 14, 30]] + \
                      [f'demand_rolling_std_{window}' for window in [7, 14, 30]] + \
                      ['Store ID_encoded', 'Product ID_encoded', 'Category_encoded']
        
        # Remove rows with NaN values (from lag features)
        df_clean = df_with_lags.dropna(subset=all_features + ['Demand'])
        
        # Split data
        split_date = df_clean['Date'].max() - pd.DateOffset(months=3)
        train_data = df_clean[df_clean['Date'] <= split_date]
        test_data = df_clean[df_clean['Date'] > split_date]
        
        X_train = train_data[all_features]
        y_train = train_data['Demand']
        X_test = test_data[all_features]
        y_test = test_data['Demand']
        
        # Use a portable sklearn fallback where the LightGBM native runtime is unavailable.
        model_name = 'LightGBM' if lgb is not None else 'HistGradientBoosting'
        lgb_model = (lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=6, random_state=42, verbose=-1)
                     if lgb is not None else
                     HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1,
                                                   max_depth=6, random_state=42))
        
        lgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lgb_model.predict(X_test)
        
        # Calculate metrics
        train_scales = train_data.sort_values('Date').groupby(['Store ID', 'Product ID'])['Demand'].apply(
            lambda values: values.diff().abs().mean()).dropna()
        naive_scale = train_scales.mean()
        mase = np.mean(np.abs(y_test.values - y_pred)) / naive_scale if naive_scale else np.nan
        naive_pred = test_data['demand_lag_7'].to_numpy()
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        smape = self.calculate_smape(y_test.values, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['lightgbm'] = {
            'model': lgb_model,
            'model_name': model_name,
            'seasonal_naive_mae': naive_mae,
            'seasonal_naive_rmse': naive_rmse,
            'seasonal_naive_mase': naive_mae / naive_scale if naive_scale else np.nan,
            'seasonal_naive_smape': self.calculate_smape(y_test.values, naive_pred),
            'predictions': y_pred,
            'actual': y_test.values,
            'test_keys': test_data[['Date', 'Store ID', 'Product ID']].reset_index(drop=True),
            'mase': mase,
            'smape': smape,
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"{model_name} Results: MASE={mase:.3f}, sMAPE={smape:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}; observed-history lag-7 naive MAE={naive_mae:.3f}")
        print("✅ LightGBM forecasting completed!")
    
    def lstm_forecasting(self):
        """Implement LSTM deep learning forecasting (simplified version)."""
        print("\n🧠 Implementing Simplified Time Series Forecasting...")
        
        # Since TensorFlow has compatibility issues, we'll implement a simplified approach
        # using linear regression with lag features as a proxy for LSTM
        
        from sklearn.linear_model import LinearRegression
        
        # Prepare data with multiple lag features
        def prepare_sequence_data(data, sequence_length=14):
            sequences = []
            targets = []
            
            for (store, product), group in data.groupby(['Store ID', 'Product ID']):
                if len(group) < sequence_length + 1:
                    continue
                
                demand_series = group['Demand'].values
                
                for i in range(len(demand_series) - sequence_length):
                    sequences.append(demand_series[i:i+sequence_length])
                    targets.append(demand_series[i+sequence_length])
            
            return np.array(sequences), np.array(targets)
        
        # Prepare data
        df_reset = self.df.reset_index()
        sequences, targets = prepare_sequence_data(df_reset)
        
        if len(sequences) == 0:
            print("Insufficient data for sequence-based forecasting")
            return
        
        # Split data
        split_idx = int(len(sequences) * 0.8)
        X_train_seq = sequences[:split_idx]
        y_train_seq = targets[:split_idx]
        X_test_seq = sequences[split_idx:]
        y_test_seq = targets[split_idx:]
        
        # Use Linear Regression as a simplified approach
        model = LinearRegression()
        model.fit(X_train_seq, y_train_seq)
        
        # Make predictions
        y_pred_seq = model.predict(X_test_seq)
        
        # Calculate metrics
        mase = self.calculate_mase(y_test_seq, y_pred_seq, y_train_seq)
        smape = self.calculate_smape(y_test_seq, y_pred_seq)
        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred_seq))
        mae = mean_absolute_error(y_test_seq, y_pred_seq)
        
        self.models['sequence_model'] = {
            'model': model,
            'predictions': y_pred_seq,
            'actual': y_test_seq,
            'mase': mase,
            'smape': smape,
            'rmse': rmse,
            'mae': mae
        }
        
        print(f"Sequence Model Results: MASE={mase:.3f}, sMAPE={smape:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
        print("✅ Sequence-based forecasting completed!")
    
    def hierarchical_reconciliation(self):
        """Aggregate bottom-level predictions into coherent daily totals."""
        if 'lightgbm' not in self.models:
            raise ValueError('Fit bottom-level LightGBM forecasts first')
        result = self.models['lightgbm']
        bottom = result['test_keys'].copy()
        bottom['Predicted'] = result['predictions']
        bottom['Actual'] = result['actual']
        levels = {
            'bottom': bottom,
            'store': bottom.groupby(['Date', 'Store ID'], as_index=False)[['Predicted', 'Actual']].sum(),
            'product': bottom.groupby(['Date', 'Product ID'], as_index=False)[['Predicted', 'Actual']].sum(),
            'total': bottom.groupby('Date', as_index=False)[['Predicted', 'Actual']].sum(),
        }
        self.forecasts['reconciled'] = levels
        return levels
    
    def evaluate_models(self):
        """Evaluate all models and compare performance."""
        print("\n📊 Evaluating Models...")
        
        evaluation_results = {}
        
        for model_name, model_results in self.models.items():
            if isinstance(model_results, dict) and 'mase' in model_results:
                evaluation_results[model_results.get('model_name', model_name)] = {
                    'MASE': model_results['mase'],
                    'sMAPE': model_results['smape'],
                    'RMSE': model_results.get('rmse', np.nan),
                    'MAE': model_results.get('mae', np.nan)
                }
                if 'seasonal_naive_mae' in model_results:
                    evaluation_results['Observed-history lag-7 naive'] = {
                        'MASE': model_results['seasonal_naive_mase'],
                        'sMAPE': model_results['seasonal_naive_smape'],
                        'RMSE': model_results['seasonal_naive_rmse'],
                        'MAE': model_results['seasonal_naive_mae'],
                    }
        
        # Create comparison table
        comparison_df = pd.DataFrame(evaluation_results).T
        
        print("\n📈 Model Performance Comparison:")
        print(comparison_df.round(3))
        
        # Visualize model comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        comparison_df['MASE'].plot(kind='bar', color='skyblue')
        plt.title('MASE Comparison')
        plt.ylabel('MASE')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        comparison_df['sMAPE'].plot(kind='bar', color='lightcoral')
        plt.title('sMAPE Comparison')
        plt.ylabel('sMAPE (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        comparison_df['RMSE'].plot(kind='bar', color='lightgreen')
        plt.title('RMSE Comparison')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        comparison_df['MAE'].plot(kind='bar', color='orange')
        plt.title('MAE Comparison')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.metrics = evaluation_results
        print("✅ Model evaluation completed!")
    
    def safety_stock_calculator(self, service_level=0.95):
        """Calculate safety stock for inventory planning."""
        print(f"\n📦 Calculating Safety Stock (Service Level: {service_level*100}%)...")
        
        # Z-score for service level
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        
        safety_stocks = {}
        
        # Calculate safety stock for each Store-SKU combination
        for (store, product), group in self.df.groupby(['Store ID', 'Product ID']):
            if len(group) < 30:  # Need sufficient data
                continue
            
            # Calculate forecast errors (using naive forecast as proxy)
            demand_values = group['Demand'].values
            naive_forecast = np.roll(demand_values, 1)
            forecast_errors = demand_values[1:] - naive_forecast[1:]
            
            # Standard deviation of forecast errors
            error_std = np.std(forecast_errors)
            
            # Safety stock = Z * σ(L)
            safety_stock = z_score * error_std
            
            safety_stocks[f"{store}_{product}"] = {
                'store': store,
                'product': product,
                'safety_stock': safety_stock,
                'error_std': error_std,
                'service_level': service_level
            }
        
        # Create safety stock summary
        safety_stock_df = pd.DataFrame([
            {
                'Store': ss['store'],
                'Product': ss['product'],
                'Safety_Stock': ss['safety_stock'],
                'Error_Std': ss['error_std']
            }
            for ss in safety_stocks.values()
        ])
        
        # Top 20 products by safety stock
        top_safety_stocks = safety_stock_df.nlargest(20, 'Safety_Stock')
        
        print("\n📋 Top 20 Products by Safety Stock:")
        print(top_safety_stocks)
        
        # Visualize safety stock distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(safety_stock_df['Safety_Stock'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Safety Stock Distribution')
        plt.xlabel('Safety Stock')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        safety_stock_df.boxplot(column='Safety_Stock')
        plt.title('Safety Stock Box Plot')
        plt.ylabel('Safety Stock')
        
        plt.subplot(2, 2, 3)
        top_safety_stocks.plot(x='Product', y='Safety_Stock', kind='bar', color='lightcoral')
        plt.title('Top 20 Products by Safety Stock')
        plt.xlabel('Product')
        plt.ylabel('Safety Stock')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        safety_by_store = safety_stock_df.groupby('Store')['Safety_Stock'].sum().sort_values(ascending=False)
        safety_by_store.plot(kind='bar', color='lightgreen')
        plt.title('Total Safety Stock by Store')
        plt.xlabel('Store')
        plt.ylabel('Total Safety Stock')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/safety_stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.safety_stocks = safety_stocks
        
        # Save safety stock recommendations
        safety_stock_df.to_csv('safety_stock_recommendations.csv', index=False)
        print("✅ Safety stock calculation completed!")
        print("📁 Safety stock recommendations saved to 'safety_stock_recommendations.csv'")
    
    def create_forecast_visualizations(self):
        """Create comprehensive forecast visualizations."""
        print("\n📊 Creating Forecast Visualizations...")
        
        # 1. Actual vs Forecast plots
        if 'lightgbm' in self.models:
            lgb_results = self.models['lightgbm']
            
            plt.figure(figsize=(15, 10))
            
            # Sample predictions for visualization
            sample_size = min(1000, len(lgb_results['actual']))
            actual_sample = lgb_results['actual'][:sample_size]
            pred_sample = lgb_results['predictions'][:sample_size]
            
            plt.subplot(2, 2, 1)
            plt.scatter(actual_sample, pred_sample, alpha=0.5, color='blue')
            plt.plot([actual_sample.min(), actual_sample.max()], 
                    [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
            plt.xlabel('Actual Demand')
            plt.ylabel('Predicted Demand')
            plt.title('Actual vs Predicted Demand (LightGBM)')
            
            plt.subplot(2, 2, 2)
            residuals = actual_sample - pred_sample
            plt.scatter(pred_sample, residuals, alpha=0.5, color='green')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Demand')
            plt.ylabel('Residuals')
            plt.title('Residual Plot (LightGBM)')
            
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=30, alpha=0.7, color='purple')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            
            plt.subplot(2, 2, 4)
            time_series_sample = np.arange(len(actual_sample))
            plt.plot(time_series_sample, actual_sample, label='Actual', alpha=0.7)
            plt.plot(time_series_sample, pred_sample, label='Predicted', alpha=0.7)
            plt.xlabel('Time Index')
            plt.ylabel('Demand')
            plt.title('Time Series: Actual vs Predicted')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/forecast_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Model performance comparison
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            metrics_df['MASE'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('MASE Comparison')
            axes[0, 0].set_ylabel('MASE')
            
            metrics_df['sMAPE'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
            axes[0, 1].set_title('sMAPE Comparison')
            axes[0, 1].set_ylabel('sMAPE (%)')
            
            metrics_df['RMSE'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
            axes[1, 0].set_title('RMSE Comparison')
            axes[1, 0].set_ylabel('RMSE')
            
            metrics_df['MAE'].plot(kind='bar', ax=axes[1, 1], color='orange')
            axes[1, 1].set_title('MAE Comparison')
            axes[1, 1].set_ylabel('MAE')
            
            plt.tight_layout()
            plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Forecast visualizations completed!")
    
    def export_results(self):
        """Export forecasts and results to CSV files."""
        print("\n💾 Exporting Results...")
        
        # Export model metrics
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            metrics_df.to_csv('model_performance_metrics.csv')
            print("📁 Model performance metrics saved to 'model_performance_metrics.csv'")
        
        # Export forecasts (if available)
        if 'lightgbm' in self.models:
            lgb_results = self.models['lightgbm']
            forecast_df = pd.DataFrame({
                'Actual': lgb_results['actual'],
                'Predicted': lgb_results['predictions'],
                'Residual': lgb_results['actual'] - lgb_results['predictions']
            })
            forecast_df.to_csv('forecast_results.csv', index=False)
            print("📁 Forecast results saved to 'forecast_results.csv'")
        
        print("✅ Results export completed!")
    
    def run_complete_pipeline(self):
        """Run the complete forecasting pipeline."""
        print("🚀 Starting Complete Retail Demand Forecasting Pipeline...")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Preprocessing
        self.preprocess_data()
        
        # Step 4: Forecasting models
        self.arima_forecasting()
        self.lightgbm_forecasting()
        # The legacy sequence demonstration splits pooled SKU windows by row,
        # so it is excluded from validated model comparisons.
        
        # Step 5: Hierarchical reconciliation
        self.hierarchical_reconciliation()
        
        # Step 6: Evaluation
        self.evaluate_models()
        
        # Safety-stock recommendations require a validated lead-time model and
        # service-level assumptions, so the legacy calculation is excluded.

        # Step 8: Visualizations
        self.create_forecast_visualizations()
        
        # Step 9: Export results
        self.export_results()
        
        print("\n🎉 Complete Pipeline Finished Successfully!")
        print("=" * 60)
        print("📁 All visualizations saved in 'visualizations/' folder")
        print("📁 Results exported to CSV files")


def main():
    """Main execution function."""
    # Initialize the forecaster
    forecaster = RetailDemandForecaster('archive (2)/sales_data.csv')
    
    # Run the complete pipeline
    forecaster.run_complete_pipeline()


if __name__ == "__main__":
    main()
