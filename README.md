# Beak-TA3
Advanced hybrid AI-methods for mineral predictive mapping

⚠ **Please read all sections carefully to setup your local environment and work with the package**. 
The provided code can run on Windows, Mac and Linux and Docker.

## 1. Repository status
Last update: **06-01-2025**

**What's new?**
- Updated installation routines
- Updated examples
- Full SOM support
- Full BNN support

**Ongoing tasks**
- Docker integration to USGS

A backup of the **older** version of the main branch can be found [here](https://github.com/DARPA-CRITICALMAAS/beak-ta3/tree/backup_main_2024-01-16_last_before_package_template_integration).

If you build the environment before **06-01-2025**, please rebuild with the current updates. 
If you set-up the environment, please **always** use the main branch for the initial setup ❗

## 2. Repository structure
The repository is structured as following:

```
├── beak-ta3/
│   └── docs/                   # documentations
│       └── ...
│   └── examples                # examples how to run selected workflows and prediction methods
│       └── ...
│   ├── releases/               # notebooks for final CMA releases to run locally
│       └── ...
│   └── setup/                
│       └── /
│           ├── docker/         # installation routine for Docker
│           ├── unix/           # installation routine fo Linux and MacOS     
│           └── win/            # installation routine for Windows     
│   └── src/                
│       └── beak/
│           ├── data/           # data folder
│               └── ...  
│           ├── evaluation/     # module for calculating evaluation metrics
│           ├── experimental/   # former utilities module for first phase events
│           ├── integration/    
│               └── statmagic   # call functions for integration of SOM and BNN into StatMaGIC 
│           ├── methods/        # algorithms for SOM and BNN predictions
│               ├── bnn         
│               └── som         
│           ├── models          # place for model definitions used in first phase
│           └── utilities       # helper functions and modules for second phase
```
## 3. Installation
### Prerequisites
All contributing developers/users need git, and a copy of the repository.
Clone the repository with:
```
git clone <https://github.com/DARPA-CRITICALMAAS/beak-ta3.git>
```
### Local desktop installation for Linux, MacOS and Windows
1. Install a conda environment on your machine
2. Execute the `setup.sh` script located in the `unix` or `win` folder, respectively

The Conda environment for both Docker and Conda installations is called **beak-ta3.**<br>

### Docker installation
Docker is **recommended** as it containerizes the whole development environment, making sure it stays identical across different developers and operating systems. Using a container also keeps your own computer clean of all dependencies.

1. Install Docker
2. Execute the `build.sh` script to build the container. The image is called `beak-ta3:latest`
3. Execute the `run.sh` script to run the container

Under Windows, either use GIT or the command-line-interface. 
For the latter, add `bash` as prefix to the script names. 

Using the `run.sh`, the container will be removed automatically after stopping. 

To avoid this behaviour, go to the root directory of the cloned repository and start the container manually from here: 
```bash
docker run -it --name beak-ta3 -p 8888:8888 -v $(pwd):/beak-ta3 beak-ta3:latest /bin/bash
```
The Docker installation uses Poetry and does not have an environment to be activated.
After creating the container, `bash` will be activated.<p>

The installation is now ready to be attached to your IDE.
**For VSCode**
1. Start the Docker container
2. Start VSCode
3. Select the "Attach to running container" option
4. Select the **beak-ta3** container
<p>
**For PyCharm**
1. Start Docker
2. Add a new interpreter based on the `beak-ta3:latest` image

**Note** that your local repository clone gets automatically mounted into the container. This means that:

- The container has a folder that redirects to the local repository called **beak-ta3**
- The repository in your computer's filesystem and in the container are exactly the same


## 4. Check installation
**Local desktop installation**
1. Activate the environment with `conda activate beak-ta3`
2. Check the installation by executing `conda list` to see all installed environments.<p>
**Docker installation**
1. Run the container using the provided commands above
2. Check the installation with `poetry show` or `poetry run pip list` in the terminal

If the **beak** package is listed, the installation was successfull.
Because the repository behaves as a **local** python package 🐍, all functions can be accessed like so:

```python
# Import
from beak.module.submodule import function

# Call
function(arguments)
```

**Data** 💾 located in the `../src/beak/data` folder can be accessed by:
```python
# Import files module
from importlib.resources import files

# Set Path
file_path = files("beak.data") / "subfolder" / "file_name.ext"
```

## 5. For developers
### Information
Thanks to [MTRI](https://github.com/DARPA-CRITICALMAAS/mtri-packagetemplate), we can use the **repository code** like a **package**. 
To install everything from scratch, follow the instructions on the linked page.
We have updated the process by replacing the `setup.py` with a `pyproject.toml` file, which enables to install the 
environment in multiple ways.<p>

The provided installation methods contain the `pip install -e .` execution, which is necessary for treating the
code as local package. If you are going to do a complete manual installation, you may need to execute this command
at the very end within your Conda or Docker installation **after setting up the environment**.<p>

There are some bottlenecks regarding the installation of packages, mainly due to different versions on the 
repositories, for  
- platform (Unix, Windows, Docker) and
- platform architecture (x86, ARM)

Particularly, `gdal` and `somoclu` packages may raise some issues if installed from [PyPi](https://pypi.org/). 
These are not part of the `pyproject.toml` environment requirements and installed separately from within the `dockerfile`.
If possible, stick to one of the Conda installations, which are based on [Conda-Forge](https://conda-forge.org/).
 
### Documentation
To build the documentation, the `sphinx` package is used. It currently works only in Linux/Docker. The provided template `builddocs.bash` may not work for everyone directly. The documentation files can be generated by running the `builddocs.bash` file from within the top-level folder of the package.

Here is how to solve common `command not found` or `encoding` issues (Windows, Linux and Mac systems have different file endings that may cause erros when executing scripts in the terminal).

**How to solve the `Command not found` error:**
1. Add the path with the `builddocs.bash` file to the `PATH` variables:
2. `export PATH=$PATH:/beak-ta3/`

**How to solve the `/bin/bash^M: bad interpreter: No such file or directory` error:**
1. Use a text editor like [Notepad++](https://notepad-plus-plus.org/) or even VSCode
2. Change the line ending to **Unix (LF)**
3. Save the file

Alternative:
1. Set the preferences for new documents to **Unix (LF)**
2. Copy/paste the file content
3. Save as another/overwrite the original file
