"""
Explainability Module for Soil Moisture Analysis
Provides model interpretation, visualization, and explanation tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable
import pandas as pd
from pathlib import Path
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInterpreter:
    """
    Interprets machine learning models for soil moisture prediction
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_gradients(self, inputs: torch.Tensor, target_class: int = 0) -> torch.Tensor:
        """Compute gradients using backpropagation"""
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        
        if len(outputs.shape) > 1:
            loss = outputs[:, target_class].sum()
        else:
            loss = outputs.sum()
            
        self.model.zero_grad()
        loss.backward()
        
        return inputs.grad.data
    
    def integrated_gradients(self, inputs: torch.Tensor, baseline: Optional[torch.Tensor] = None,
                           steps: int = 50) -> torch.Tensor:
        """Compute Integrated Gradients attribution"""
        if baseline is None:
            baseline = torch.zeros_like(inputs)
            
        alphas = torch.linspace(0, 1, steps).to(self.device)
        integrated_grads = torch.zeros_like(inputs)
        
        for alpha in alphas:
            interpolated_input = baseline + alpha * (inputs - baseline)
            grads = self.compute_gradients(interpolated_input)
            integrated_grads += grads
            
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (inputs - baseline)
        
        return integrated_grads

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for soil moisture prediction models
    """
    
    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(100)]
        
    def analyze_band_importance(self, s1_data: np.ndarray, s2_data: np.ndarray, 
                              predictions: np.ndarray) -> Dict[str, float]:
        """Analyze importance of different satellite bands"""
        correlations = {}
        
        s1_bands = ['VV', 'VH']
        for i, band in enumerate(s1_bands):
            if i < s1_data.shape[0]:
                band_data = s1_data[i].flatten()
                pred_data = predictions.flatten()
                corr, _ = pearsonr(band_data, pred_data)
                correlations[f'S1_{band}'] = abs(corr)
        
        s2_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        for i, band in enumerate(s2_bands):
            if i < s2_data.shape[0]:
                band_data = s2_data[i].flatten()
                pred_data = predictions.flatten()
                corr, _ = pearsonr(band_data, pred_data)
                correlations[f'S2_{band}'] = abs(corr)
                
        return correlations

class VisualizationTools:
    """
    Creates visualizations for model explanations and predictions
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_attribution_map(self, attribution: np.ndarray, original_image: Optional[np.ndarray] = None,
                           title: str = "Attribution Map", save_path: Optional[str] = None):
        """Plot attribution map overlaid on original image"""
        fig, axes = plt.subplots(1, 2 if original_image is not None else 1, 
                                figsize=self.figsize)
        
        if original_image is not None:
            if not isinstance(axes, np.ndarray):
                axes = [axes]
                
            if len(original_image.shape) == 3:
                axes[0].imshow(np.transpose(original_image[:3], (1, 2, 0)))
            else:
                axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            im = axes[1].imshow(attribution, cmap='RdBu_r', vmin=-np.max(np.abs(attribution)), 
                               vmax=np.max(np.abs(attribution)))
            axes[1].set_title(title)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
        else:
            im = plt.imshow(attribution, cmap='RdBu_r', vmin=-np.max(np.abs(attribution)), 
                           vmax=np.max(np.abs(attribution)))
            plt.title(title)
            plt.axis('off')
            plt.colorbar(im)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_vs_actual(self, predictions: np.ndarray, actuals: np.ndarray,
                                 title: str = "Predictions vs Actual", save_path: Optional[str] = None):
        """Plot predictions vs actual values with metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        ax1.scatter(actuals.flatten(), predictions.flatten(), alpha=0.5)
        ax1.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Soil Moisture')
        ax1.set_ylabel('Predicted Soil Moisture')
        ax1.set_title('Predictions vs Actual')
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        corr, _ = pearsonr(actuals.flatten(), predictions.flatten())
        
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nCorr = {corr:.3f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        residuals = predictions - actuals
        ax2.scatter(actuals.flatten(), residuals.flatten(), alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Actual Soil Moisture')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spatial_predictions(self, predictions: np.ndarray, extent: Optional[List[float]] = None,
                               title: str = "Soil Moisture Predictions", save_path: Optional[str] = None):
        """Plot spatial distribution of soil moisture predictions"""
        plt.figure(figsize=self.figsize)
        
        colors = ['#8B4513', '#DEB887', '#90EE90', '#006400']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('soil_moisture', colors, N=n_bins)
        
        if extent:
            im = plt.imshow(predictions, cmap=cmap, extent=extent, origin='lower')
        else:
            im = plt.imshow(predictions, cmap=cmap)
        
        plt.colorbar(im, label='Soil Moisture Content')
        plt.title(title)
        plt.xlabel('Longitude' if extent else 'X (pixels)')
        plt.ylabel('Latitude' if extent else 'Y (pixels)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class UncertaintyQuantifier:
    """
    Quantifies prediction uncertainty using various methods
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        
    def monte_carlo_dropout(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using Monte Carlo Dropout"""
        self.model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        self.model.eval()
        
        return mean_pred, uncertainty

def calculate_model_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive model performance metrics"""
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()
    
    metrics = {
        'MSE': mean_squared_error(actuals_flat, predictions_flat),
        'RMSE': np.sqrt(mean_squared_error(actuals_flat, predictions_flat)),
        'MAE': mean_absolute_error(actuals_flat, predictions_flat),
        'R2': r2_score(actuals_flat, predictions_flat),
        'Pearson_Correlation': pearsonr(actuals_flat, predictions_flat)[0],
        'Spearman_Correlation': spearmanr(actuals_flat, predictions_flat)[0],
        'MAPE': np.mean(np.abs((actuals_flat - predictions_flat) / actuals_flat)) * 100
    }
    
    return metrics

__all__ = [
    'ModelInterpreter',
    'FeatureImportanceAnalyzer', 
    'VisualizationTools',
    'UncertaintyQuantifier',
    'calculate_model_metrics'
]

