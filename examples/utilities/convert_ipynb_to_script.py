import os
from pathlib import Path
from beak.experimental.utilities.io import create_file_list
from nbconvert.nbconvertapp import NbConvertApp

def convert_notebook_to_script(notebook_path):
    notebook_path = str(notebook_path)
    nbconvert_app = NbConvertApp()
    nbconvert_app.export_format = "script"
    nbconvert_app.initialize(argv=[notebook_path])
    nbconvert_app.start()


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    notebook_list = create_file_list(folder=Path(current_dir), extensions=[".ipynb"], recursive=True)

    for notebook in notebook_list:
        script_file = notebook.stem + ".py"
        
        print(f"Convert {notebook}...")
        convert_notebook_to_script(notebook)
            