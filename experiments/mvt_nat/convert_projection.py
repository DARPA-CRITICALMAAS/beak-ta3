import utilities as utils
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from time import sleep


# CONSTANTS
INPUT_ROOT_FOLDER = "data/LAWLEY22-DATACUBE-EXPORT/MVT_PREFERRED/"
OUTPUT_ROOT_FOLDER = "data/LAWLEY22-DATACUBE-EXPORT/MVT_PREFERRED_REPROJECTED/"
TARGET_EPSG = 3857
TARGET_RESOLUTION = 5000


def load_files(folder):
    files, _ = utils.load_rasters(folder)
    return files


def reproject_file(file, input_folder, output_folder, target_epsg, target_resolution):
    raster = utils.load_raster(file)
    out_file = output_folder / file.relative_to(Path(input_folder))
    utils.check_path(out_file.parent)
    out_array, out_meta = utils.reproject_raster(raster, target_epsg, target_resolution)
    utils.save_raster(
        out_file,
        out_array,
        target_epsg,
        out_meta["height"],
        out_meta["width"],
        raster.nodata,
        out_meta["transform"],
    )


def main(
    input_folder,
    output_folder,
    target_epsg,
    target_resolution,
    n_workers=mp.cpu_count(),
):
    # Show selected folder
    print(f"Selected folder: {input_folder}")

    # Get all folders in the root folder
    folders, _ = utils.create_folder_list(Path(input_folder))
    print(f"Total of folders found: {len(folders)}")

    # Load rasters for each folder
    file_list = []

    with mp.Pool() as pool:
        results = pool.map(load_files, folders)

    for result in results:
        file_list.extend(result)

    # Show results
    print(f"Files loaded: {len(file_list)}")

    # Set args list
    args_list = [
        (file, input_folder, output_folder, target_epsg, target_resolution)
        for file in file_list
    ]

    # Run multiprocessing
    pool = mp.Pool(n_workers)
    with tqdm(total=len(args_list), desc="Processing files") as pbar:
        for _ in pool.starmap(reproject_file, args_list):
            pbar.update(1)
            sleep(0.1)


if __name__ == "__main__":
    main(INPUT_ROOT_FOLDER, OUTPUT_ROOT_FOLDER, TARGET_EPSG, TARGET_RESOLUTION)
