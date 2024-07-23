#!/usr/bin/env python
# coding: utf-8

# Models and utils
import os
import argparse
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

def write_args_to_file(file_path, **kwargs):
  with open(file_path, "w") as file:
    json.dump(kwargs, file, indent=4)
    file.close()

def main(args):
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

    # Set model name and folder
    current_dir = Path(os.path.dirname(__file__)).resolve()
    os.chdir(current_dir)

    MODEL_NAME = "F" + str(len(file_list)) + "_X" + str(args.som_x) + "_Y" + str(args.som_y) + "_E" + str(args.epochs) + "_CMAX" + str(args.kmeans_max) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    MODEL_FOLDER = Path.cwd() / "models" / MODEL / MODEL_NAME
    print(MODEL_FOLDER)

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

    args_path = MODEL_FOLDER / "args.json"
    write_args_to_file(file_path=args_path,
                       som_x=args.som_x,
                       som_y=args.som_y,
                       epochs=args.epochs,
                       kmeans=args.kmeans,
                       kmeans_init=args.kmeans_init,
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

    # Args
    args.output_folder = str(MODEL_FOLDER) + "/" + "exports"
    check_path(args.output_folder)

    args.output_file_somspace = args.output_folder + "/" + "result_som.txt"
    args.outgeofile = args.output_folder + "/" + "result_geo.txt"
    args.output_file_geospace = args.outgeofile

    from beak.methods.som.argsSOM import Args
    args.input_file = Args().create_list_from_file(str(file_path))
    args.geotiff_input=args.input_file
    args.label_geotiff_file = Args().create_list_from_file(str(label_data_file_path))

    # Run SOM
    import beak.methods.som.do_nextsomcore_save_results as dnsr
    import beak.methods.som.move_to_subfolder as mts
    import warnings

    args.output_folder = str(args.output_folder)
    mts.remove_som_results(args.output_folder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dnsr.run_SOM(args)

    # Plotting
    import beak.methods.som.plot_som_results as plot

    # Load cluster dictionary
    loaded_cluster_list = plot.load_cluster_dictionary(args.output_folder)

    # Plot and save the Davies-Bouldin Index vs Number of Clusters
    plot.plot_davies_bouldin(loaded_cluster_list, args.output_folder)

    import beak.methods.som.argsPlot
    import beak.methods.som.plot_som_results as plot
    import beak.methods.som.move_to_subfolder as mts

    argsP = beak.methods.som.argsPlot.Args()

    argsP.outsomfile = args.output_file_somspace
    argsP.som_x = args.som_x
    argsP.som_y = args.som_y
    argsP.input_file = args.input_file
    argsP.dir = args.output_folder
    argsP.grid_type = 'rectangular'
    argsP.redraw = True
    argsP.outgeofile = args.output_file_geospace
    argsP.dataType = 'grid'
    argsP.noDataValue = ' -9999'

    plot.run_plotting_script(argsP)

    subfolder_name = "plots"
    images, labels = mts.move_figures(args.output_folder, subfolder_name)

    # Delete results_geo.txt due to its gigantic size for regional assessments
    os.remove(Path(args.output_folder) / "result_geo.txt")


# Add arguments to parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOM")
    parser.add_argument("--som_x", type=int, default=50)                    # X dimension of generated SOM
    parser.add_argument("--som_y", type=int, default=50)                    # Y dimension of generated SOM
    parser.add_argument("--epochs", type=int, default=10)                   # Number of epochs to run
    parser.add_argument("--kmeans", type=bool, default=True)                # Run k-means clustering
    parser.add_argument("--kmeans_init", type=int, default=5)               # Number of initializations
    parser.add_argument("--kmeans_min", type=int, default=20)               # Minimum number of clusters
    parser.add_argument("--kmeans_max", type=int, default=50)               # Maximum number of clusters
    parser.add_argument("--neighborhood", type=str, default="gaussian")     # Neightborhood shape
    parser.add_argument("--std_coeff", type=float, default=0.5)             # Gaussian Coefficient
    parser.add_argument("--maptype", type=str, default="toroid")            # SOM Map type
    parser.add_argument("--initialcodebook", type=str, default=None)        # Codebook vectors path (if)
    parser.add_argument("--radius0", type=float, default=0)                 # Initial neighborhood size
    parser.add_argument("--radiusN", type=float, default=1)                 # Final neighborhood size
    parser.add_argument("--radiuscooling", type=str, default="linear")      # Neighborhood size decrease
    parser.add_argument("--scalecooling", type=str, default="linear")       # Learning scale decrease
    parser.add_argument("--scale0", type=float, default=0.1)                # Initial learning rate
    parser.add_argument("--scaleN", type=float, default=0.01)               # Final learning rate
    parser.add_argument("--initialization", type=str, default="random")     # SOM initialization
    parser.add_argument("--gridtype", type=str, default="rectangular")      # SOM grid shape
    parser.add_argument("--label", type=bool, default=True)                 # Labels (if)
    parser.add_argument("--normalized", type=bool, default=False)           # Normalize the output units

    args = parser.parse_args()
    main(args)
