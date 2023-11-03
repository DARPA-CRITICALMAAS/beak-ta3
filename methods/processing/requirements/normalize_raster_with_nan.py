import numpy as np


# Function to normalize a raster
def normalize_raster_with_nan(raster):
    # Find the minimum and maximum values while ignoring NaNs
    min_val = np.nanmin(raster)
    max_val = np.nanmax(raster)

    # Normalize the raster data while handling NaNs
    normalized_raster = np.where(np.isnan(raster), np.nan, (raster - min_val) / (max_val - min_val))
    
    return normalized_raster