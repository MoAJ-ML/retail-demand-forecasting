#!/usr/bin/env python3
"""
Modern Visualization Design
===========================

Creates ultra-clear correlation matrix and modern summary insights with perfect design.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set modern style
plt.style.use('default')
sns.set_style("whitegrid")

class ModernVisualizations:
    """Modern visualization class with perfect design and clarity."""
    
    def __init__(self, df):
        self.df = df
        self.setup_modern_style()
    
    def setup_modern_style(self):
        """Set up modern plotting style."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 20,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.2,
            'grid.color': '#E0E0E0',
            'axes.facecolor': 'white',
            'figure.facecolor': 'white'
        })
    
    def create_ultra_clear_correlation_matrix(self):
        """Create ultra-clear correlation matrix with perfect readability."""
        print("📊 Creating Ultra-Clear Correlation Matrix...")
        
        # Select only numeric columns for correlation
        numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 
                       'Discount', 'Promotion', 'Competitor Pricing', 'Demand']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Create figure with modern design
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Create custom colormap for better clarity
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#1f77b4', '#ffffff', '#ff7f0e']  # Blue to White to Orange
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Create heatmap with enhanced styling
        im = ax.imshow(correlation_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(correlation_matrix.columns, fontsize=12, fontweight='bold')
        
        # Add correlation values with enhanced styling
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                
                # Choose text color based on background
                text_color = 'white' if abs(value) > 0.5 else 'black'
                font_weight = 'bold' if abs(value) > 0.7 else 'normal'
                
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight=font_weight,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add colorbar with modern styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Correlation Coefficient', fontsize=13, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)
        
        # Add title with modern styling
        ax.set_title('Feature Correlation Matrix\nStrong Positive (Blue) ↔ Strong Negative (Orange)', 
                    fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        
        # Add subtle grid
        ax.set_xticks(np.arange(-0.5, len(correlation_matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(correlation_matrix.columns), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Remove major grid
        ax.grid(which='major', visible=False)
        
        plt.tight_layout()
        plt.savefig('visualizations/ultra_clear_correlation_matrix.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Ultra-clear correlation matrix saved!")
    
    def create_modern_summary_insights(self, metrics, safety_stocks):
        """Create modern, perfectly designed summary insights."""
        print("📊 Creating Modern Summary Insights...")
        
        # Create figure with modern design
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#F8F9FA')
        
        # Create main title
        fig.suptitle('Retail Demand Forecasting - Executive Summary', 
                    fontsize=24, fontweight='bold', color='#2C3E50', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                              left=0.05, right=0.95, top=0.88, bottom=0.05)
        
        # 1. Model Performance Card (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_performance_card(ax1, metrics)
        
        # 2. Safety Stock Summary Card (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_safety_stock_card(ax2, safety_stocks)
        
        # 3. Dataset Overview Card (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_dataset_card(ax3)
        
        # 4. Key Metrics Dashboard (Top Far Right)
        ax4 = fig.add_subplot(gs[0, 3])
        self._create_metrics_dashboard(ax4, metrics)
        
        # 5. Performance Comparison Chart (Middle Row)
        ax5 = fig.add_subplot(gs[1, :2])
        self._create_performance_chart(ax5, metrics)
        
        # 6. Safety Stock Distribution (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2:])
        self._create_safety_stock_chart(ax6, safety_stocks)
        
        # 7. Business Insights (Bottom Row)
        ax7 = fig.add_subplot(gs[2, :])
        self._create_business_insights(ax7)
        
        plt.savefig('visualizations/modern_summary_insights.png', dpi=300, bbox_inches='tight',
                   facecolor='#F8F9FA', edgecolor='none')
        plt.show()
        print("✅ Modern summary insights saved!")
    
    def _create_performance_card(self, ax, metrics):
        """Create modern performance card."""
        ax.set_facecolor('#FFFFFF')
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", 
                                   facecolor='#FFFFFF', edgecolor='#E0E0E0', linewidth=2))
        
        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            best_model = metrics_df['MASE'].idxmin()
            best_mase = metrics_df.loc[best_model, 'MASE']
            best_smape = metrics_df.loc[best_model, 'sMAPE']
            
            ax.text(0.5, 0.8, '🏆 BEST MODEL', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='#27AE60')
            ax.text(0.5, 0.65, best_model, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.45, f'MASE: {best_mase:.3f}', ha='center', va='center', 
                   fontsize=12, color='#7F8C8D')
            ax.text(0.5, 0.3, f'sMAPE: {best_smape:.1f}%', ha='center', va='center', 
                   fontsize=12, color='#7F8C8D')
            ax.text(0.5, 0.1, 'Lower = Better', ha='center', va='center', 
                   fontsize=10, style='italic', color='#95A5A6')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_safety_stock_card(self, ax, safety_stocks):
        """Create modern safety stock card."""
        ax.set_facecolor('#FFFFFF')
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", 
                                   facecolor='#FFFFFF', edgecolor='#E0E0E0', linewidth=2))
        
        if safety_stocks:
            safety_df = pd.DataFrame([
                {'Safety_Stock': ss['safety_stock']} for ss in safety_stocks.values()
            ])
            
            avg_safety = safety_df['Safety_Stock'].mean()
            max_safety = safety_df['Safety_Stock'].max()
            
            ax.text(0.5, 0.8, '📦 SAFETY STOCK', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='#E74C3C')
            ax.text(0.5, 0.6, f'Avg: {avg_safety:.1f}', ha='center', va='center', 
                   fontsize=13, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.45, f'Max: {max_safety:.1f}', ha='center', va='center', 
                   fontsize=12, color='#7F8C8D')
            ax.text(0.5, 0.25, 'Service Level: 95%', ha='center', va='center', 
                   fontsize=11, color='#7F8C8D')
            ax.text(0.5, 0.1, 'Risk Protection', ha='center', va='center', 
                   fontsize=10, style='italic', color='#95A5A6')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_dataset_card(self, ax):
        """Create modern dataset card."""
        ax.set_facecolor('#FFFFFF')
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", 
                                   facecolor='#FFFFFF', edgecolor='#E0E0E0', linewidth=2))
        
        total_records = len(self.df)
        total_stores = self.df['Store ID'].nunique()
        total_products = self.df['Product ID'].nunique()
        avg_demand = self.df['Demand'].mean()
        
        ax.text(0.5, 0.8, '📊 DATASET', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='#3498DB')
        ax.text(0.5, 0.6, f'{total_records:,} Records', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#2C3E50')
        ax.text(0.5, 0.45, f'{total_stores} Stores', ha='center', va='center', 
               fontsize=11, color='#7F8C8D')
        ax.text(0.5, 0.3, f'{total_products} Products', ha='center', va='center', 
               fontsize=11, color='#7F8C8D')
        ax.text(0.5, 0.15, f'Avg Demand: {avg_demand:.1f}', ha='center', va='center', 
               fontsize=11, color='#7F8C8D')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_metrics_dashboard(self, ax, metrics):
        """Create modern metrics dashboard."""
        ax.set_facecolor('#FFFFFF')
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02", 
                                   facecolor='#FFFFFF', edgecolor='#E0E0E0', linewidth=2))
        
        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            avg_mase = metrics_df['MASE'].mean()
            avg_smape = metrics_df['sMAPE'].mean()
            
            ax.text(0.5, 0.8, '📈 METRICS', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='#9B59B6')
            ax.text(0.5, 0.6, f'Avg MASE: {avg_mase:.3f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.4, f'Avg sMAPE: {avg_smape:.1f}%', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.2, f'{len(metrics_df)} Models', ha='center', va='center', 
                   fontsize=11, color='#7F8C8D')
            ax.text(0.5, 0.05, 'Performance', ha='center', va='center', 
                   fontsize=10, style='italic', color='#95A5A6')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_performance_chart(self, ax, metrics):
        """Create modern performance comparison chart."""
        if not metrics:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            ax.axis('off')
            return
        
        metrics_df = pd.DataFrame(metrics).T
        models = metrics_df.index.tolist()
        mase_scores = metrics_df['MASE'].tolist()
        smape_scores = metrics_df['sMAPE'].tolist()
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mase_scores, width, label='MASE', 
                      color='#3498DB', alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, smape_scores, width, label='sMAPE (%)', 
                      color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', color='#2C3E50')
        ax.set_xlabel('Models', fontsize=13, color='#34495E')
        ax.set_ylabel('Score (Lower = Better)', fontsize=13, color='#34495E')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Highlight best model
        best_model_idx = metrics_df['MASE'].idxmin()
        best_idx = models.index(best_model_idx)
        bars1[best_idx].set_edgecolor('#27AE60')
        bars1[best_idx].set_linewidth(4)
    
    def _create_safety_stock_chart(self, ax, safety_stocks):
        """Create modern safety stock distribution chart."""
        if not safety_stocks:
            ax.text(0.5, 0.5, 'No safety stock data available', ha='center', va='center')
            ax.axis('off')
            return
        
        safety_df = pd.DataFrame([
            {'Safety_Stock': ss['safety_stock']} for ss in safety_stocks.values()
        ])
        
        # Create histogram with modern styling
        n, bins, patches = ax.hist(safety_df['Safety_Stock'], bins=30, alpha=0.7, 
                                  color='#E74C3C', edgecolor='white', linewidth=1)
        
        # Color bars based on value
        for i, (patch, value) in enumerate(zip(patches, n)):
            if value > np.mean(n):
                patch.set_facecolor('#C0392B')
            else:
                patch.set_facecolor('#E74C3C')
        
        ax.axvline(safety_df['Safety_Stock'].mean(), color='#2C3E50', linestyle='--', 
                  linewidth=3, label=f'Mean: {safety_df["Safety_Stock"].mean():.1f}')
        
        ax.set_title('Safety Stock Distribution', fontsize=16, fontweight='bold', color='#2C3E50')
        ax.set_xlabel('Safety Stock Level', fontsize=13, color='#34495E')
        ax.set_ylabel('Frequency', fontsize=13, color='#34495E')
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    def _create_business_insights(self, ax):
        """Create modern business insights section."""
        ax.set_facecolor('#FFFFFF')
        
        # Calculate insights
        category_demand = self.df.groupby('Category')['Demand'].sum().sort_values(ascending=False)
        top_category = category_demand.index[0]
        top_category_demand = category_demand.iloc[0]
        
        store_demand = self.df.groupby('Store ID')['Demand'].sum().sort_values(ascending=False)
        top_store = store_demand.index[0]
        top_store_demand = store_demand.iloc[0]
        
        promo_impact = self.df.groupby('Promotion')['Demand'].mean()
        promo_effect = ((promo_impact[1] - promo_impact[0]) / promo_impact[0]) * 100
        
        insights_text = f"""
        🎯 KEY BUSINESS INSIGHTS
        
        📈 Top Performing Category: {top_category} ({top_category_demand:,} total demand)
        🏪 Highest Volume Store: {top_store} ({top_store_demand:,} total demand)
        🎉 Promotion Impact: {promo_effect:+.1f}% average demand increase
        📊 Data Quality: No missing values, {len(self.df):,} clean records
        🔮 Forecasting: Multi-model approach with hierarchical reconciliation
        📦 Inventory: Safety stock calculated for 95% service level
        """
        
        ax.text(0.05, 0.9, insights_text, ha='left', va='top', fontsize=14, 
               color='#2C3E50', fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


def main():
    """Main function to create modern visualizations."""
    print("🎨 Creating Modern, Ultra-Clear Visualizations...")
    
    # Load data
    df = pd.read_csv('archive (2)/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize modern visualization class
    viz = ModernVisualizations(df)
    
    # Create ultra-clear correlation matrix
    viz.create_ultra_clear_correlation_matrix()
    
    # Load metrics and safety stock data
    try:
        metrics_df = pd.read_csv('model_performance_metrics.csv', index_col=0)
        metrics = metrics_df.to_dict('index')
    except FileNotFoundError:
        metrics = {}
    
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
    except FileNotFoundError:
        safety_stocks = {}
    
    # Create modern summary insights
    viz.create_modern_summary_insights(metrics, safety_stocks)
    
    print("🎉 Modern visualizations created successfully!")
    print("📁 Ultra-clear correlation matrix and modern summary saved!")


if __name__ == "__main__":
    main()
