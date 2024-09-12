import os
import sys
import subprocess
from pathlib import Path


def run_script(script_path, *args):
    # Run the script using the absolute path
    print(f"Running {script_path}")
    process = subprocess.Popen(
        ["python", script_path] + list(args),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()


# Set current file director to working directory
current_dir = Path(os.path.dirname(__file__)).resolve()
os.chdir(current_dir)

# Single list
# -------------
# List of scripts to run, path can take subfolders: subfoler/example.py
list_to_call = []

# Run
for file_path, som_args in list_to_call:
    run_script(file_path, *som_args)


# Looped list
# -----------
loop_items = [
    #"MORPH_MAMBA",
    "BASELINE_BISON",
    #"JITTER_JELLYFISH",
]

for item in loop_items:
    list_to_call = [
        (
            "02_som.py",
            ["--model_config", item, "--epochs", "10", 
             "--som_x", "42", "--som_y", "42", 
             "--kmeans_min","45", "--kmeans_max","50"]
        )
    ]

    # Run
    for file_path, som_args in list_to_call:
        run_script(file_path, *som_args)
