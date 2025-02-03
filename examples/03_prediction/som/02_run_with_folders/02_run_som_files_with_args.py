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

# List of scripts to run, path can take subfolders: subfoler/example.py
list_to_call = [
    ("01_som_example_call.py",
     ["--epochs", "50", "--som_x", "10", "--som_y", "10"]),
]

# Run
for file_path, som_args in list_to_call:
    run_script(file_path, *som_args)
