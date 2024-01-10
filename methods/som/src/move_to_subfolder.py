import shutil
import os
import glob


def move_som_results(file_path, subfolder_name):

    file_patterns = ["*som.*","*geo.*", "RunStats.txt","cluster.dictionary","db_score.png"]
    destination_path = file_path + "/" + subfolder_name + "/"

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for file_pattern in file_patterns:
        # Use glob to get all files with the specified pattern
        matching_files = glob.glob(os.path.join(file_path, file_pattern))

        # Move each matching file to the destination folder and overwrite existing files if necessary
        for source_file in matching_files:
            file_name = os.path.basename(source_file)
            destination_file_path = os.path.join(destination_path, file_name)

            # If the file already exists in the destination folder, delete it first
            if os.path.exists(destination_file_path):
                os.remove(destination_file_path)

            # Move the file to the destination folder
            shutil.move(source_file, destination_file_path)


def move_figures(file_path, subfolder_name):

    file_patterns = ["geoplot_*.png", "somplot_*.png", "boxplot_*.png", "db_score.png"]
    destination_path = file_path + "/" + subfolder_name + "/"

    # Lists to store matching files with their corresponding destination paths
    all_figs = []
    all_figs_lable = []

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    else:
        # Delete all files in the destination folder
        files_to_delete = glob.glob(os.path.join(destination_path, "*"))
        for file_to_delete in files_to_delete:
            os.remove(file_to_delete)

    for file_pattern in file_patterns:
        # Use glob to get all files with the specified pattern
        matching_files = glob.glob(os.path.join(file_path, file_pattern))

        # Add matching files and their corresponding destination paths to the lists
        all_figs.extend([os.path.join(destination_path, os.path.basename(file)) for file in matching_files])
        all_figs_lable.extend([os.path.basename(file) for file in matching_files])

        # Move each matching file to the destination folder and overwrite existing files if necessary
        for source_file in matching_files:
            file_name = os.path.basename(source_file)
            destination_file_path = os.path.join(destination_path, file_name)

            # If the file already exists in the destination folder, delete it first
            if os.path.exists(destination_file_path):
                os.remove(destination_file_path)

            # Move the file to the destination folder
            shutil.move(source_file, destination_file_path)

    return all_figs, all_figs_lable