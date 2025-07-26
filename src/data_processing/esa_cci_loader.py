"""
ESA CCI soil moisture data loading and processing for London area.
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import griddata
from pathlib import Path

class ESACCILoader:
    """Load and process ESA CCI soil moisture data for validation."""
    
    def __init__(self):
        self.london_bounds = {
            'lon_min': -0.5, 'lon_max': 0.5,
            'lat_min': 51.2, 'lat_max': 51.8
        }
    
    def load_netcdf_data(self, filename):
        """Load soil moisture data from NetCDF file."""
        ds = Dataset(filename, mode='r')
        
        print("Available variables:", list(ds.variables.keys()))
        
        # Extract data
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        sm = ds.variables['sm'][:]
        
        print(f"Soil moisture shape: {sm.shape}")
        print(f"Lat range: {lat.min()} to {lat.max()}")
        print(f"Lon range: {lon.min()} to {lon.max()}")
        
        return lat, lon, sm, ds
    
    def crop_to_london(self, lat, lon, sm):
        """Crop data to London bounding box."""
        bounds = self.london_bounds
        
        # Find indices
        lat_inds = np.where((lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))[0]
        lon_inds = np.where((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']))[0]
        
        # Crop data
        sm_london = sm[0, lat_inds.min():lat_inds.max()+1, lon_inds.min():lon_inds.max()+1]
        lat_london = lat[lat_inds.min():lat_inds.max()+1]
        lon_london = lon[lon_inds.min():lon_inds.max()+1]
        
        print(f"Cropped SM shape: {sm_london.shape}")
        return sm_london, lat_london, lon_london
    
    def resample_to_target_grid(self, sm_london, lat_london, lon_london, target_size=1000):
        """Resample to high-resolution grid."""
        bounds = self.london_bounds
        
        # Create coarse grid
        lon_grid, lat_grid = np.meshgrid(lon_london, lat_london)
        
        # Flatten coords and values
        points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
        values = sm_london.ravel()
        
        # Target high-res grid
        target_x = np.linspace(bounds['lon_min'], bounds['lon_max'], target_size)
        target_y = np.linspace(bounds['lat_min'], bounds['lat_max'], target_size)
        target_xx, target_yy = np.meshgrid(target_x, target_y)
        
        # Interpolate
        sm_resampled = griddata(points, values, (target_xx, target_yy), method='linear')
        
        print(f"Resampled to: {sm_resampled.shape}")
        return sm_resampled, target_xx, target_yy
    
    def visualize_data(self, data, title, extent=None):
        """Visualize soil moisture data."""
        if extent is None:
            extent = [self.london_bounds['lon_min'], self.london_bounds['lon_max'],
                     self.london_bounds['lat_min'], self.london_bounds['lat_max']]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='YlGnBu', extent=extent, origin='lower')
        plt.title(title)
        plt.colorbar(label="Moisture (m³/m³)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()
