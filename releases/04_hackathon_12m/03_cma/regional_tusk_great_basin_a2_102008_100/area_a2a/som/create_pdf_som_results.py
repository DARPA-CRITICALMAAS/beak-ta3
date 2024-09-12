from beak.methods.som.pngs_to_pdf_loop import pngs_to_pdf
# Files
import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files

def args():
    BASE_PATH = files("beak")

    CMA = "regional_tusk_great_basin_a2_102008_100"
    MODEL = "Hacking_Hamster_PP"  # Add all your models here
    RUNS = [
        "F28_X50_Y50_E15_CMAX50_20240821-220116",
        "F28_X30_Y30_E50_CMAX30_20240821-184210",
        "F28_X30_Y30_E15_CMAX50_20240821-200828",
        "F28_X30_Y30_E15_CMAX30_20240821-182329"
        ]
    
    # User-provided names for each RUN
    RUN_NAMES = [
        "50x50 E15 CMAX50",
        "30x30 E50 CMAX30",
        "30x30 E50 CMAX30",
        "30x30 E15 CMAX50"
    ]

    # User-provided names for title
    TITLE_NAME = "Cluster results " + MODEL    

    OUTPUT_NAME = "results_" + MODEL + ".pdf"
    ROOT_PATH = BASE_PATH / ".." / ".." / "experiments" / "04_hackathon_12m_related" / "03_cma"
    folder_paths = [ROOT_PATH / CMA / "som" / "models" / MODEL / run / "exports" / "plots" for run in RUNS]
    output_pdf_path = ROOT_PATH / CMA / "som" / "models" / OUTPUT_NAME
    
    number_of_columns = len(folder_paths)  # Number of columns will be equal to the number of models

    return [folder_paths, RUN_NAMES, TITLE_NAME, output_pdf_path, number_of_columns]


# Call the function
[folder_paths, run_names, title_name, output_pdf_path, number_of_columns] = args()

pngs_to_pdf(folder_paths, run_names, title_name, output_pdf_path, number_of_columns)