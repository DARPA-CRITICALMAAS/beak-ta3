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
     #"BASELINE_BISON_PP",
     #"JITTER_JELLYFISH_PP",
     #"HACKING_HAMSTER_PP",
     "MORPH_MAMBA_PP"
]

for item in loop_items:
    list_to_call = [
        (
            "02_som.py",
            ["--model_config", item, 
             "--epochs", "15", 
             "--som_x", "30", "--som_y", "30",
             "--kmeans_max","50", 
             "--kmeans_min","30",
             "--initialization", "pca"]
        )
    ]

    # Run
    for file_path, som_args in list_to_call:
        run_script(file_path, *som_args)
