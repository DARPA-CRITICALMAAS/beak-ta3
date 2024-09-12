from beak.methods.som.pngs_to_pdf import pngs_to_pdf

# Files
import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files

def args():
    BASE_PATH = files("beak")

    CMA = "regional_tusk_great_basin_a2_102008_100"
    MODEL = "JITTER_JELLYFISH_PP"
    RUN = "F29_X30_Y30_E15_CMAX30_20240821-165041"

    OUTPUT_NAME = "result_" + MODEL + "_" + RUN + ".pdf"
    ROOT_PATH = BASE_PATH / ".." / ".." / "experiments" / "04_hackathon_12m_related" / "03_cma"
    folder_path = ROOT_PATH / CMA / "som" / "models" / MODEL / RUN / "exports" / "plots"
    output_pdf_path = ROOT_PATH / CMA / "som" / "models" / OUTPUT_NAME
    
    number_of_images_per_row = 3

    return [folder_path, output_pdf_path, number_of_images_per_row]


# Call the function
[folder_path, output_pdf_path, number_of_images_per_row] = args()

pngs_to_pdf(folder_path, output_pdf_path, number_of_images_per_row)