import os
import sys
import subprocess
from pathlib import Path

def run_script(script_path):
    # Run the script using the absolute path
    print(f"Running {script_path}")
    process = subprocess.Popen(
        ["python", script_path],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()


# Set current file director to working directory
current_dir = Path(os.path.dirname(__file__)).resolve()
os.chdir(current_dir)

# List of scripts to run
scripts = [
    "regional_102008_500_mama_nico_upmidwest/som/02_som_baseline_bison.py",
    "regional_102008_500_mama_nico_upmidwest/som/02_som_jitter_jellyfish.py",
    "regional_102008_500_mvt_ceus/som/02_som_jitter_jellyfish.py",
    "regional_102008_500_poco_southwest/som/02_som_baseline_bison.py",
    "regional_102008_500_poco_southwest/som/02_som_jitter_jellyfish.py",
    "regional_102008_50_poco_southwest_nm/som/02_som_jitter_jellyfish_pp.py",
    "regional_102008_50_mvt_southmid_cont/som/02_som_baseline_bison_pp.py",
    "regional_102008_50_mvt_southmid_cont/som/02_som_jitter_jellyfish.py",
]

# Run
for script_path in scripts:
    run_script(script_path)
