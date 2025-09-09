#!/usr/bin/env python3
"""
Clean Forecast Analysis Visualization
====================================

Creates focused, clear forecast analysis plots without unnecessary elements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_clean_forecast_analysis():
    """Create clean, focused forecast analysis."""
    print("📊 Creating Clean Forecast Analysis...")
    
    # Load forecast results
    try:
        forecast_df = pd.read_csv('forecast_results.csv')
        print(f"Loaded {len(forecast_df)} forecast results")
    except FileNotFoundError:
        print("Forecast results file not found")
        return
    
    # Set up clean plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (15, 10),
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
    
    # Sample data for cleaner visualization (first 2000 points)
    sample_size = min(2000, len(forecast_df))
    sample_df = forecast_df.head(sample_size)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Forecast Performance Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Actual vs Predicted Scatter Plot
    axes[0, 0].scatter(sample_df['Actual'], sample_df['Predicted'], 
                      alpha=0.6, s=20, color='#2E86AB', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(sample_df['Actual'].min(), sample_df['Predicted'].min())
    max_val = max(sample_df['Actual'].max(), sample_df['Predicted'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                   label='Perfect Prediction')
    
    # Calculate and display R²
    r2 = r2_score(sample_df['Actual'], sample_df['Predicted'])
    axes[0, 0].text(0.05, 0.95, f'R² Score: {r2:.3f}', 
                   transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    axes[0, 0].set_title('Actual vs Predicted Demand', fontweight='bold')
    axes[0, 0].set_xlabel('Actual Demand')
    axes[0, 0].set_ylabel('Predicted Demand')
    axes[0, 0].legend()
    
    # 2. Residuals Analysis
    residuals = sample_df['Residual']
    
    axes[0, 1].scatter(sample_df['Predicted'], residuals, alpha=0.6, s=20, 
                      color='#A23B72', edgecolors='white', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residuals Analysis', fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Demand')
    axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
    
    # Add residual statistics
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    axes[0, 1].text(0.05, 0.95, f'Mean: {residual_mean:.2f}\nStd: {residual_std:.2f}', 
                   transform=axes[0, 1].transAxes, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 3. Residuals Distribution
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='#F18F01', 
                   edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(residual_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {residual_mean:.2f}')
    axes[1, 0].axvline(residual_mean + residual_std, color='orange', linestyle=':', linewidth=2,
                      label=f'+1 Std: {residual_mean + residual_std:.2f}')
    axes[1, 0].axvline(residual_mean - residual_std, color='orange', linestyle=':', linewidth=2,
                      label=f'-1 Std: {residual_mean - residual_std:.2f}')
    axes[1, 0].set_title('Residuals Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Time Series Sample (first 100 points for clarity)
    time_sample = min(100, len(sample_df))
    time_indices = range(time_sample)
    
    axes[1, 1].plot(time_indices, sample_df['Actual'].head(time_sample), 
                   label='Actual', linewidth=2.5, color='#2E86AB', alpha=0.8)
    axes[1, 1].plot(time_indices, sample_df['Predicted'].head(time_sample), 
                   label='Predicted', linewidth=2.5, color='#A23B72', alpha=0.8)
    axes[1, 1].set_title(f'Time Series Comparison\n(First {time_sample} Points)', fontweight='bold')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Demand')
    axes[1, 1].legend()
    
    # Add performance metrics
    mae = mean_absolute_error(sample_df['Actual'], sample_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(sample_df['Actual'], sample_df['Predicted']))
    
    axes[1, 1].text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                   transform=axes[1, 1].transAxes, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('visualizations/clean_forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Clean forecast analysis saved!")


def create_performance_summary():
    """Create a clean performance summary."""
    print("📊 Creating Performance Summary...")
    
    try:
        metrics_df = pd.read_csv('model_performance_metrics.csv', index_col=0)
    except FileNotFoundError:
        print("Model performance metrics file not found")
        return
    
    # Create clean performance comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for visualization
    models = metrics_df.index.tolist()
    mase_scores = metrics_df['MASE'].tolist()
    smape_scores = metrics_df['sMAPE'].tolist()
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mase_scores, width, label='MASE', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, smape_scores, width, label='sMAPE (%)', color='#A23B72', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Model Performance Comparison\n(Lower Values = Better Performance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Highlight best model
    best_model_idx = metrics_df['MASE'].idxmin()
    best_idx = models.index(best_model_idx)
    ax.get_children()[best_idx*2].set_edgecolor('red')
    ax.get_children()[best_idx*2].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('visualizations/clean_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Clean performance summary saved!")


def main():
    """Main function to create clean visualizations."""
    print("🎨 Creating Clean, Focused Visualizations...")
    
    create_clean_forecast_analysis()
    create_performance_summary()
    
    print("🎉 Clean visualizations created successfully!")
    print("📁 All visualizations saved in 'visualizations/' folder")


if __name__ == "__main__":
    main()
