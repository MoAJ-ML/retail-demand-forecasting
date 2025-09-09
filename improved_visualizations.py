#!/usr/bin/env python3
"""
Improved Visualization Module for Retail Demand Forecasting
=========================================================

Enhanced visualizations with better clarity, readability, and focus on key insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class ImprovedVisualizations:
    """Enhanced visualization class with improved clarity and readability."""
    
    def __init__(self, df):
        self.df = df
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up consistent plotting style for all visualizations."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_enhanced_eda_overview(self):
        """Create enhanced EDA overview with better clarity."""
        print("📊 Creating Enhanced EDA Overview...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Retail Demand Analysis - Key Insights', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Demand Distribution (simplified)
        axes[0, 0].hist(self.df['Demand'], bins=50, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('Demand Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Daily Demand')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['Demand'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["Demand"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Demand by Category (top 5 only)
        category_demand = self.df.groupby('Category')['Demand'].sum().sort_values(ascending=False).head(5)
        bars = axes[0, 1].bar(range(len(category_demand)), category_demand.values, 
                             color=['#A23B72', '#F18F01', '#C73E1D', '#2E86AB', '#7209B7'])
        axes[0, 1].set_title('Total Demand by Category (Top 5)', fontweight='bold')
        axes[0, 1].set_xlabel('Product Category')
        axes[0, 1].set_ylabel('Total Demand')
        axes[0, 1].set_xticks(range(len(category_demand)))
        axes[0, 1].set_xticklabels(category_demand.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Demand by Store (top 5 only)
        store_demand = self.df.groupby('Store ID')['Demand'].sum().sort_values(ascending=False).head(5)
        bars = axes[0, 2].bar(range(len(store_demand)), store_demand.values,
                             color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
        axes[0, 2].set_title('Total Demand by Store (Top 5)', fontweight='bold')
        axes[0, 2].set_xlabel('Store ID')
        axes[0, 2].set_ylabel('Total Demand')
        axes[0, 2].set_xticks(range(len(store_demand)))
        axes[0, 2].set_xticklabels(store_demand.index)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Daily Demand Trend (simplified - monthly aggregation)
        daily_demand = self.df.groupby('Date')['Demand'].sum()
        monthly_demand = daily_demand.resample('M').sum()
        axes[1, 0].plot(monthly_demand.index, monthly_demand.values, 
                       marker='o', linewidth=3, markersize=8, color='#2E86AB')
        axes[1, 0].set_title('Monthly Demand Trend', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Total Demand')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Price vs Demand Scatter (simplified)
        sample_data = self.df.sample(n=min(5000, len(self.df)), random_state=42)
        scatter = axes[1, 1].scatter(sample_data['Price'], sample_data['Demand'], 
                                   alpha=0.6, s=20, color='#A23B72')
        axes[1, 1].set_title('Price vs Demand Relationship', fontweight='bold')
        axes[1, 1].set_xlabel('Price ($)')
        axes[1, 1].set_ylabel('Demand')
        
        # Add correlation coefficient
        correlation = sample_data['Price'].corr(sample_data['Demand'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Promotion Impact
        promo_impact = self.df.groupby('Promotion')['Demand'].mean()
        bars = axes[1, 2].bar(['No Promotion', 'Promotion'], promo_impact.values,
                             color=['#C73E1D', '#2E86AB'])
        axes[1, 2].set_title('Average Demand: Promotion Impact', fontweight='bold')
        axes[1, 2].set_ylabel('Average Demand')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Enhanced EDA overview saved!")
    
    def create_enhanced_correlation_matrix(self):
        """Create enhanced correlation matrix with better readability."""
        print("📊 Creating Enhanced Correlation Matrix...")
        
        # Select only numeric columns for correlation
        numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 
                       'Discount', 'Promotion', 'Competitor Pricing', 'Demand']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create heatmap with better styling
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        
        plt.title('Feature Correlation Matrix\n(Red = Positive, Blue = Negative)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Enhanced correlation matrix saved!")
    
    def create_enhanced_model_comparison(self, metrics):
        """Create enhanced model comparison with better clarity."""
        print("📊 Creating Enhanced Model Comparison...")
        
        if not metrics:
            print("No metrics available for comparison")
            return
        
        metrics_df = pd.DataFrame(metrics).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        # Define colors for models
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # MASE Comparison
        bars = axes[0, 0].bar(range(len(metrics_df)), metrics_df['MASE'], 
                             color=colors[:len(metrics_df)])
        axes[0, 0].set_title('MASE Comparison\n(Lower is Better)', fontweight='bold')
        axes[0, 0].set_ylabel('MASE Score')
        axes[0, 0].set_xticks(range(len(metrics_df)))
        axes[0, 0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # sMAPE Comparison
        bars = axes[0, 1].bar(range(len(metrics_df)), metrics_df['sMAPE'], 
                             color=colors[:len(metrics_df)])
        axes[0, 1].set_title('sMAPE Comparison\n(Lower is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('sMAPE (%)')
        axes[0, 1].set_xticks(range(len(metrics_df)))
        axes[0, 1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # RMSE Comparison
        bars = axes[1, 0].bar(range(len(metrics_df)), metrics_df['RMSE'], 
                             color=colors[:len(metrics_df)])
        axes[1, 0].set_title('RMSE Comparison\n(Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_xticks(range(len(metrics_df)))
        axes[1, 0].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE Comparison
        bars = axes[1, 1].bar(range(len(metrics_df)), metrics_df['MAE'], 
                             color=colors[:len(metrics_df)])
        axes[1, 1].set_title('MAE Comparison\n(Lower is Better)', fontweight='bold')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_xticks(range(len(metrics_df)))
        axes[1, 1].set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Enhanced model comparison saved!")
    
    def create_enhanced_forecast_analysis(self, model_results):
        """Create enhanced forecast analysis with better clarity."""
        print("📊 Creating Enhanced Forecast Analysis...")
        
        if 'lightgbm' not in model_results:
            print("LightGBM results not available for forecast analysis")
            return
        
        lgb_results = model_results['lightgbm']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Forecast Analysis - LightGBM Model', fontsize=20, fontweight='bold', y=0.98)
        
        # Sample data for visualization (first 1000 points)
        sample_size = min(1000, len(lgb_results['actual']))
        actual_sample = lgb_results['actual'][:sample_size]
        pred_sample = lgb_results['predictions'][:sample_size]
        
        # 1. Actual vs Predicted Scatter
        axes[0, 0].scatter(actual_sample, pred_sample, alpha=0.6, s=30, color='#2E86AB')
        
        # Perfect prediction line
        min_val = min(actual_sample.min(), pred_sample.min())
        max_val = max(actual_sample.max(), pred_sample.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_title('Actual vs Predicted Demand', fontweight='bold')
        axes[0, 0].set_xlabel('Actual Demand')
        axes[0, 0].set_ylabel('Predicted Demand')
        axes[0, 0].legend()
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(actual_sample, pred_sample)
        axes[0, 0].text(0.05, 0.95, f'R² Score: {r2:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals Plot
        residuals = actual_sample - pred_sample
        axes[0, 1].scatter(pred_sample, residuals, alpha=0.6, s=30, color='#A23B72')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residuals Plot', fontweight='bold')
        axes[0, 1].set_xlabel('Predicted Demand')
        axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
        
        # 3. Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {residuals.mean():.2f}')
        axes[1, 0].set_title('Residuals Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Time Series Sample (first 200 points)
        time_sample_size = min(200, len(actual_sample))
        time_indices = range(time_sample_size)
        
        axes[1, 1].plot(time_indices, actual_sample[:time_sample_size], 
                       label='Actual', linewidth=2, color='#2E86AB', alpha=0.8)
        axes[1, 1].plot(time_indices, pred_sample[:time_sample_size], 
                       label='Predicted', linewidth=2, color='#A23B72', alpha=0.8)
        axes[1, 1].set_title('Time Series: Actual vs Predicted\n(First 200 Points)', fontweight='bold')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Demand')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_forecast_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Enhanced forecast analysis saved!")
    
    def create_enhanced_safety_stock_analysis(self, safety_stocks):
        """Create enhanced safety stock analysis with better clarity."""
        print("📊 Creating Enhanced Safety Stock Analysis...")
        
        if not safety_stocks:
            print("No safety stock data available")
            return
        
        # Convert to DataFrame
        safety_df = pd.DataFrame([
            {
                'Store': ss['store'],
                'Product': ss['product'],
                'Safety_Stock': ss['safety_stock'],
                'Error_Std': ss['error_std']
            }
            for ss in safety_stocks.values()
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Safety Stock Analysis - Inventory Planning', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Safety Stock Distribution
        axes[0, 0].hist(safety_df['Safety_Stock'], bins=50, alpha=0.7, color='#2E86AB', 
                       edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(safety_df['Safety_Stock'].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {safety_df["Safety_Stock"].mean():.1f}')
        axes[0, 0].set_title('Safety Stock Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Safety Stock Level')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Top 10 Products by Safety Stock
        top_10 = safety_df.nlargest(10, 'Safety_Stock')
        bars = axes[0, 1].barh(range(len(top_10)), top_10['Safety_Stock'], 
                              color='#A23B72')
        axes[0, 1].set_title('Top 10 Products by Safety Stock', fontweight='bold')
        axes[0, 1].set_xlabel('Safety Stock Level')
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels([f"{row['Store']}-{row['Product']}" for _, row in top_10.iterrows()])
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 1].text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                           f'{width:.1f}', ha='left', va='center', fontweight='bold')
        
        # 3. Safety Stock by Store
        store_safety = safety_df.groupby('Store')['Safety_Stock'].sum().sort_values(ascending=False)
        bars = axes[1, 0].bar(range(len(store_safety)), store_safety.values,
                             color='#F18F01')
        axes[1, 0].set_title('Total Safety Stock by Store', fontweight='bold')
        axes[1, 0].set_xlabel('Store ID')
        axes[1, 0].set_ylabel('Total Safety Stock')
        axes[1, 0].set_xticks(range(len(store_safety)))
        axes[1, 0].set_xticklabels(store_safety.index)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Safety Stock vs Error Std Scatter
        sample_safety = safety_df.sample(n=min(500, len(safety_df)), random_state=42)
        axes[1, 1].scatter(sample_safety['Error_Std'], sample_safety['Safety_Stock'], 
                          alpha=0.6, s=30, color='#C73E1D')
        axes[1, 1].set_title('Safety Stock vs Forecast Error Std', fontweight='bold')
        axes[1, 1].set_xlabel('Forecast Error Standard Deviation')
        axes[1, 1].set_ylabel('Safety Stock Level')
        
        # Add correlation
        correlation = sample_safety['Error_Std'].corr(sample_safety['Safety_Stock'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_safety_stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Enhanced safety stock analysis saved!")
    
    def create_summary_insights(self, metrics, safety_stocks):
        """Create a summary insights visualization."""
        print("📊 Creating Summary Insights...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Key Business Insights Summary', fontsize=20, fontweight='bold', y=1.02)
        
        # 1. Best Model Performance
        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            best_model = metrics_df['MASE'].idxmin()
            best_mase = metrics_df.loc[best_model, 'MASE']
            best_smape = metrics_df.loc[best_model, 'sMAPE']
            
            axes[0].text(0.5, 0.7, f'Best Model: {best_model}', 
                        ha='center', va='center', fontsize=16, fontweight='bold',
                        transform=axes[0].transAxes)
            axes[0].text(0.5, 0.5, f'MASE: {best_mase:.3f}', 
                        ha='center', va='center', fontsize=14,
                        transform=axes[0].transAxes)
            axes[0].text(0.5, 0.3, f'sMAPE: {best_smape:.1f}%', 
                        ha='center', va='center', fontsize=14,
                        transform=axes[0].transAxes)
            axes[0].set_title('Model Performance', fontweight='bold')
            axes[0].axis('off')
        
        # 2. Safety Stock Summary
        if safety_stocks:
            safety_df = pd.DataFrame([
                {'Safety_Stock': ss['safety_stock']} for ss in safety_stocks.values()
            ])
            
            avg_safety = safety_df['Safety_Stock'].mean()
            max_safety = safety_df['Safety_Stock'].max()
            
            axes[1].text(0.5, 0.7, f'Average Safety Stock: {avg_safety:.1f}', 
                        ha='center', va='center', fontsize=14,
                        transform=axes[1].transAxes)
            axes[1].text(0.5, 0.5, f'Maximum Safety Stock: {max_safety:.1f}', 
                        ha='center', va='center', fontsize=14,
                        transform=axes[1].transAxes)
            axes[1].text(0.5, 0.3, f'Service Level: 95%', 
                        ha='center', va='center', fontsize=14,
                        transform=axes[1].transAxes)
            axes[1].set_title('Safety Stock Planning', fontweight='bold')
            axes[1].axis('off')
        
        # 3. Data Summary
        total_records = len(self.df)
        total_stores = self.df['Store ID'].nunique()
        total_products = self.df['Product ID'].nunique()
        avg_demand = self.df['Demand'].mean()
        
        axes[2].text(0.5, 0.8, f'Total Records: {total_records:,}', 
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes)
        axes[2].text(0.5, 0.6, f'Stores: {total_stores}', 
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes)
        axes[2].text(0.5, 0.4, f'Products: {total_products}', 
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes)
        axes[2].text(0.5, 0.2, f'Avg Daily Demand: {avg_demand:.1f}', 
                    ha='center', va='center', fontsize=14,
                    transform=axes[2].transAxes)
        axes[2].set_title('Dataset Overview', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/summary_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Summary insights saved!")


def main():
    """Main function to create all enhanced visualizations."""
    print("🎨 Creating Enhanced Visualizations...")
    
    # Load data
    df = pd.read_csv('archive (2)/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize visualization class
    viz = ImprovedVisualizations(df)
    
    # Create enhanced visualizations
    viz.create_enhanced_eda_overview()
    viz.create_enhanced_correlation_matrix()
    
    # Load model metrics if available
    try:
        metrics_df = pd.read_csv('model_performance_metrics.csv', index_col=0)
        metrics = metrics_df.to_dict('index')
        viz.create_enhanced_model_comparison(metrics)
    except FileNotFoundError:
        print("Model metrics file not found, skipping model comparison")
    
    # Load safety stock data if available
    try:
        safety_df = pd.read_csv('safety_stock_recommendations.csv')
        safety_stocks = {}
        for _, row in safety_df.iterrows():
            key = f"{row['Store']}_{row['Product']}"
            safety_stocks[key] = {
                'store': row['Store'],
                'product': row['Product'],
                'safety_stock': row['Safety_Stock'],
                'error_std': row['Error_Std']
            }
        viz.create_enhanced_safety_stock_analysis(safety_stocks)
    except FileNotFoundError:
        print("Safety stock file not found, skipping safety stock analysis")
    
    # Create summary insights
    try:
        metrics_df = pd.read_csv('model_performance_metrics.csv', index_col=0)
        metrics = metrics_df.to_dict('index')
        safety_df = pd.read_csv('safety_stock_recommendations.csv')
        safety_stocks = {}
        for _, row in safety_df.iterrows():
            key = f"{row['Store']}_{row['Product']}"
            safety_stocks[key] = {
                'store': row['Store'],
                'product': row['Product'],
                'safety_stock': row['Safety_Stock'],
                'error_std': row['Error_Std']
            }
        viz.create_summary_insights(metrics, safety_stocks)
    except FileNotFoundError:
        print("Required files not found for summary insights")
    
    print("🎉 All enhanced visualizations created successfully!")


if __name__ == "__main__":
    main()
