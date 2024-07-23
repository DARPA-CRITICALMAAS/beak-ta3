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

# Example
list_to_call_example = [
    (
        "regional_102008_500_mama_nico_upmidwest/som/02_som_jitter_jellyfish.py",
        ["--epochs", "50", "--som_x", "50", "--som_y", "50"],
    ),
]


# Loop
loop_items = [
    "04_som_goofy_gopher.py",
    "04_som_yolo_yak.py",
    "05_som_camel_case.py",
    "05_som_pretty_penguin.py"
]

for item in loop_items:
    list_to_call = [
    (
        "regional_102008_500_poco_southwest/som/" + item,
        ["--epochs", "10", "--som_x", "40", "--som_y", "40"],
    ),
]

    # Run
    for file_path, som_args in list_to_call:
        run_script(file_path, *som_args)


# Manual mode
list_to_call = [

]

for file_path, som_args in list_to_call:
    run_script(file_path, *som_args)
