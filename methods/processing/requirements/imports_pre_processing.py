import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd # GeoTiff format needed for EIS-toolkit process
from tqdm import tqdm  # Import the tqdm library for progress bars
import rasterio 
from rasterio import features, profiles, transform
from rasterio.enums import MergeAlg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages #To create and export a pdf report file
from scipy import stats  # Import the stats module from scipy
from requirements.rasterize_vector import rasterize_vector # EIS Tool-kit module and functions
from requirements.normalize_raster_with_nan import normalize_raster_with_nan #Informal function self defined