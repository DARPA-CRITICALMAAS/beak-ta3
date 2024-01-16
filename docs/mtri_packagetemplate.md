# MTRI TA4 Integration Template

Template to support integration into MTRI TA4 effort

## Installation (Linux)
>Note: Contact MTRI for Windows instructions (aplyons@mtu.edu, efvega@mtu.edu)

To install this package into an environment, follow the commands in this list.
- Optional (but recommended): Create and source a Python virtual environment
- `python3 -m venv $HOME/.virtualenvs/YOURENVIRONMENTNAME`
- `source $HOME/.virtualenvs/YOURENVIRONMENTNAME/bin/activate`
- Upgrade pip and install wheel
  - `pip install --upgrade pip`
  - `pip install wheel`
- Install your package
  - The `-e` tells the environment that source files may change, so you don't have to 
  re-run the `pip install` command after you edit your files.
  - `pip install -e .`

## Things to change
There are several references that need to be changed in this template package for use in developing your own package. 
Namely, any reference to the name `packagetemplate` needs to be changed to the name of your own package. The following 
list contains all the references to `packagetemplate`.
> Note: If you have installed the package first via `pip install -e .`, there will be many more references in 
> subdirectories. Only the references in the below list need to be changed. A new environment will need to be created 
> to install the new package in.
- In `src/packagetemplate/math/normalKLDiv.py`.
  - import statement.
- In `setup.py`.
  - Several lines. Seach for `packagetemplate` and change all found references.
- In `tests/packagetemplate/test_normalKLDiv.py`.
  - import statement.
  - Line in `test_readPackageData()` that reads local data.
- The `src/packagetemplate` directory name.
- The `tests/packagetemplate` directory name.

With all of these references changed, your new package should build successfully.

## Example usages
### Data shipped with your package
Data can be shipped with the Python package that you put together. An example can be seen at the path 
`packagetemplate/src/packagetemplate/data/testData.npy`. This data can be used in your package's source code; 
or more likely, in your validation tests. An example of how to reference the packaged data can be seen in the file
`packagetemplate/tests/packagetemplate/test_normalKLDiv.py`. For compatibility, it is useful to include the 
conditional import block at the top of the file instead of just either import:
```
import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
```
This gives you access to the `files()` function that can be used to reference the data. The testing function 
`test_readPackageData()` contains the code used to actually load the data:
```
test_array = np.load(str(files("packagetemplate.data") / "testData.npy"))
```
This line uses the `np.load()` function to load a string constructed from the `files()` call. The `files()` 
function will provide an absolute system path to the location specified as an argument. Your data file can 
then be concatenated onto the `file()` call to get an absolute path to your data.

### Relative importing within the package
Building your Python code as a package allows you to easily import parts of your codebase without worrying about 
absolute and relative paths. After installing the package into your environment with `pip install -e .`, you should 
be able to import your package from anywhere in your code. Examples can be seen in the following files:
- `src/packagetemplate/math/normalKLDiv.py`.
- `tests/packagetemplate/test_normalKLDiv.py`.


### Specifying package requirements
When building you Python code as a package, it is not strictly necessary to create a `requirements.txt` file that 
stores your project dependencies. Instead, there is a block in the `setup.py` file that takes care of dependency 
installation for you. Any dependency can be listed in the `install_requires` block of `setup.py`, and will be installed 
when you install your package with `pip install -e .`. More information can be seen in the comments of `setup.py`.

### Building documentation
Documentation can be built for your package by running the `builddocs.bash` file found in the top-level package 
directory. This script uses Python's `sphinx` library to build a series of HTML files based on the docstrings in 
your code. The HTML documentation can be viewed by running `firefox docs/build/html/index.html` from the top-level 
package directory, or double-clicking on `index.html` from a file-explorer.

