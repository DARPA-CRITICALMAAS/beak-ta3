#!/usr/bin/env python
# coding: utf-8

# # 1: Import libraries and set paths

# In[1]:


# Models and utils
import json
from pathlib import Path
from datetime import datetime
from beak.models import hack_12m_mvt
from beak.utilities.io import load_model, check_path

# SOM specific
import beak.methods.som.argsSOM as asom
from beak.methods.som.nextsomcore.nextsomcore import NxtSomCore
args = asom.Args()

# Files
import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files

# Choose model
MODEL = "BASELINE_BISON"
model = hack_12m_mvt.regional_scale_ceus[MODEL]

BASE_PATH = files("beak.data")

# Choose data path
ROOT_PATH = BASE_PATH / "PROCESSED" / "regional_102008_500_mvt_ceus"
PATH_STD_DATA = ROOT_PATH / "unified_scaled_std"
PATH_LOG_DATA = ROOT_PATH / "unified_scaled_log"
PATH_LABELS = ROOT_PATH / "labels" / "TA2_240609_HM9_MCCAFFERTY_TRAIN.tif"

model_dict, file_list, counts = load_model(
    model=model,
    folders=[PATH_STD_DATA, PATH_LOG_DATA],
    file_extensions=[".tif", ".tiff"],
    verbose=0,
)

# SOM specific
label_data_file_list = [str(PATH_LABELS)]


# # 2: Set SOM parameters and model **name** and **configuration**

# In[2]:


args.som_x = 50                    # X dimension of generated SOM
args.som_y = 50                    # Y dimension of generated SOM
args.epochs = 20                   # Number of epochs to run

args.kmeans = True                 # Run k-means clustering (True, False)
args.kmeans_init = 5               # Number of initializations
args.kmeans_min = 20               # Minimum number of k-mean clusters
args.kmeans_max = 50               # Maximum number of k-mean clusters

args.neighborhood = "gaussian"     # Shape of the neighborhood function. gaussian or bubble
args.std_coeff = 0.5               # Coefficient in the Gaussian neighborhood function
args.maptype = "toroid"            # Type of SOM (sheet, toroid)
args.initialcodebook = None        # File path of initial codebook, 2D numpy.array of float32.
args.radius0 = 0                   # Initial size of the neighborhood
args.radiusN = 1                   # Final size of the neighborhood
args.radiuscooling = "linear"      # Function that defines the decrease in the neighborhood size as the training proceeds (linear, exponential)
args.scalecooling = "linear"       # Function that defines the decrease in the learning scale as the training proceeds (linear, exponential)
args.scale0 = 0.1                  # Initial learning rate
args.scaleN = 0.01                 # Final learning rate
args.initialization = "random"     # Type of SOM initialization (random, pca)
args.gridtype = "rectangular"      # Type of SOM grid (hexagonal, rectangular)

args.label = True                  # Whether data contains label column, True or False


# In[3]:


# Set model name and folder
import os
current_dir = Path(os.path.dirname(__file__)).resolve()
os.chdir(current_dir)

MODEL_NAME = "F" + str(len(file_list)) + "_X" + str(args.som_x) + "_Y" + str(args.som_y) + "_E" + str(args.epochs) + "_CMAX" + str(args.kmeans_max) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_FOLDER = Path.cwd() / "models" / MODEL / MODEL_NAME

print(MODEL_FOLDER)


# # 3: Input data, file lists etc.

# In[4]:


check_path(MODEL_FOLDER)

# Set input files path
file_path = MODEL_FOLDER / "input_file_list.txt"
label_data_file_path = MODEL_FOLDER / "label_file_list.txt"

# Write input file paths and parameters to text files
with open(file_path, 'w') as file:
    for string in file_list:
        file.write(f"{string}\n")
    file.close()

with open(label_data_file_path, 'w') as file:
    for string in label_data_file_list:
        file.write(f"{string}\n")
    file.close()

args_path = MODEL_FOLDER / "args.txt"
def write_args_to_file(file_path, **kwargs):
  with open(file_path, "w") as f:
    json.dump(kwargs, f, indent=4)
    file.close()

args_path = MODEL_FOLDER / "args.json"
write_args_to_file(file_path=args_path,
                   som_x=args.som_x,
                   som_y=args.som_y,
                   epochs=args.epochs,
                   kmeans=args.kmeans,
                   k_means_init=args.kmeans_init,
                   kmeans_min=args.kmeans_min,
                   kmeans_max=args.kmeans_max,
                   neighborhood=args.neighborhood,
                   std_coeff=args.std_coeff,
                   maptype=args.maptype,
                   initialcodebook=args.initialcodebook,
                   radius0=args.radius0,        
                   radiusN=args.radiusN,                  
                   radiuscooling=args.radiuscooling, 
                   scalecooling=args.scalecooling,
                   scale0=args.scale0,          
                   scaleN=args.scaleN,
                   initialization=args.initialization,
                   gridtype=args.gridtype,
                   label=args.label
                   )


# In[5]:


# Args
args.output_folder = str(MODEL_FOLDER) + "/" + "exports"                             # Folder to save som dictionary and cluster dictionary
check_path(args.output_folder)

args.output_file_somspace = args.output_folder + "/" + "result_som.txt"              # Txt file for: som_x som_y b_data1 b_data2 b_dataN umatrix cluster, geospace. DO NOT CHANGE!
args.outgeofile = args.output_folder + "/" + "result_geo.txt"                        # DO NOT CHANGE!
args.output_file_geospace = args.outgeofile                                          # Text file for {X Y Z} data1 data2 dataN som_x som_y cluster b_data1 b_data2 b_dataN, geospace.
#args.label_geotiff_file = args.output_folder + "/" + "input_file_list.txt"           # GeoTiff_input file (None)


# In[6]:


args.input_file = args.create_list_from_file(file_path)
args.geotiff_input=args.input_file                                # Geotiff_input files, separated by komma, to write GeoTIF out 
                                                                  # (only first line is used to get the geotransform and projection information 
                                                                  # to set output GeoFIT geotransform and projection)
args.label_geotiff_file = args.create_list_from_file(label_data_file_path)


# # 4: Run SOM 

# Run SOM with parameters specified above and save the results. Uses NxtSomCore package to do the actual work. <p>
# Before running SOM - clean up existing files and move them to a subfolder.

# In[7]:


import beak.methods.som.do_nextsomcore_save_results as dnsr
import beak.methods.som.move_to_subfolder as mts
import warnings

mts.remove_som_results(args.output_folder)                              # move or remove existing SOM output files from previous runs into subfolder
                                                                        # mts.move_som_results(args.output_folder, "old_results")
# Run SOM
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dnsr.run_SOM(args)
    


# In[8]:


import beak.methods.som.plot_som_results as plot
from IPython.display import Image, display, clear_output

# Load cluster dictionary
loaded_cluster_list = plot.load_cluster_dictionary(args.output_folder)

# Plot and save the Davies-Bouldin Index vs Number of Clusters
plot.plot_davies_bouldin(loaded_cluster_list, args.output_folder)


# # 5: Plot results.

# Specify the parameters to plot the results and create figures. The Python script "plot_som_results.py" creates .png files of the results in som space, geospace and also creates boxplots.

# Move figures into a sub folder. If the destination folder does not exist, it is created here. All file names are stored in a list that is used in the next step to show all output figures.

# In[9]:


import beak.methods.som.argsPlot
import beak.methods.som.plot_som_results as plot
import beak.methods.som.move_to_subfolder as mts

argsP = beak.methods.som.argsPlot.Args()

argsP.outsomfile = args.output_file_somspace            # som calculation somspace output text file
argsP.som_x = args.som_x                                # som x dimension
argsP.som_y = args.som_y                                # som y dimension
argsP.input_file = args.input_file                      # input file (*.lrn)
argsP.dir = args.output_folder                          # input file (*.lrn) or directory where som.dictionary was safed to (/output/som.dictionary)
argsP.grid_type = 'rectangular'                         # grid type (square or hexa), (rectangular or hexagonal)
argsP.redraw = True                                     # True: draw all plots. False: draw only polts required for clustering.
argsP.outgeofile = args.output_file_geospace            # som geospace results txt file
argsP.dataType = 'grid'                                 # data type (scatter or grid)
argsP.noDataValue= ' -9999'                             # nodata value

plot.run_plotting_script(argsP)

subfolder_name = "plots"
images, labels = mts.move_figures(args.output_folder, subfolder_name)


# In[10]:


import os

# Delete results_geo.txt due to its gigantic size for regional assessments
os.remove(Path(args.output_folder) / "result_geo.txt")

