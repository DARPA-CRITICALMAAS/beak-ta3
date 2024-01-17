# Beak-TA3
Advanced hybrid AI-methods for mineral predictive mapping

⚠ **Please read all sections carefully to setup your local environment and work with the package**. 
The provided code can run on Windows, Mac and Linux machines, if correctly set up.

## 1. Repository status

This repository is still in development. Documentation will grow over time.

Last update: **17-01-2023**

What's new?
- MTRI packagetemplate integration

A backup of the **older** version of the main branch can be found [here](https://github.com/DARPA-CRITICALMAAS/beak-ta3/tree/backup_main_2024-01-16_last_before_package_template_integration).

If you build the environment before **11-12-2023**, please rebuild with the current updates. 
If you set-up the environment, please **always** use the main branch for the initial setup ❗

## 2. Repository structure

The repository structure has changed and modified to fit the requirements for MTRI and the provided [package template](https://github.com/DARPA-CRITICALMAAS/mtri-packagetemplate). 

The repository is structured as following:

```
├── beak-ta3/
│   ├── docs/                   # documentations
│   ├── local/                  # stuff that's not being uploaded to GitHub
│   └── src/                
│       └── beak/
│           ├── data            # data folder
│           ├── experiments/    # mineral systems and model configurations
│               └── ...         
│           ├── methods/        # code for respective ML methods
│               ├── bnn         
│               ├── hybrid         
│               └── som         
│           └── utilities       # helper functions and modules       
│   └── tests/              
│       └── beak                # test functions  
```

Since there is a function that links to the data folder to provide easy data access, **please save all data within the data folder** ❗ 

Also, the content of the **data folder** will not be synchronized with the GitHub repository 📴

The development for **SOM**, **BNN** and potential **hybridization** is accomplished on the respective branches:
- som_dev
- bnn_dev
- hybrid_dev

Depending on the development status, results will be merged to the main branch. 
Some may be created when starting the development.

## 3. Prerequisites

All contributing developers/users need git, and a copy of the repository.

```
git clone <https://github.com/DARPA-CRITICALMAAS/beak-ta3.git>
```

Alternatively, you can also you GitHub Desktop instead of Git for repo-management.

After cloning, there are two options setting up your local development environment.

1. Docker
2. Conda

Docker is recommended as it containerizes the whole development environment, making sure it stays identical across different developers and operating systems. Using a container also keeps your own computer clean of all dependencies.

If you’re not familiar with development inside a container, see [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers) and [https://code.visualstudio.com/docs/devcontainers/tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) to prepare, e.g. for VSCODE.

All code and functionality is designed to run on CPU since GPU-processing requires special preparation and environment setup depending on graphics card (driver), CUDA and TensorFlow versions.

The conda environment for both Docker and Conda installations is called **beak-ta3.** 

## 4. Environment setup

### Setting up a local environment with docker
#### Installation

Create a terminal instance from within the beak-ta3 folder. Build and run the container. Run this and every other command in the repository root unless otherwise directed.

```
docker compose up -d
```

If you need to rebuild already existing container (e.g. dependencies have been updated), run

```
docker compose up -d --build
```

#### Working with the container

Attach to the running container

```
docker attach beak-ta3-dev
```

You are now in your local development container, and all your commands in the current terminal window interact with the container.

If you want to work using the terminal, you also need to open the SHELL in advance and activate the environment:

```
bash
conda activate beak-ta3
```

**Note** that your local repository gets automatically mounted into the container. This means that:

- The container has a folder that redirects to the local repository called **beak-ta3**
- The repository in your computer's filesystem and in the container are exactly the same

**Alternatively**, if your are working with **VSCode**, you can also use the following steps to start:
1. Start the Docker container
2. Start VSCode
3. Select the "Attach to running container" option
4. Select the **beak-ta3** container

### Setting up a local conda environment
#### Installation

You can also set-up a local conda environment on your OS using the **environment.yml** from within the cloned repository folder.

Create the basic environment:
```
conda env create -f environment.yml
```

Activate it:
```
conda activate beak-ta3
```

Install additional packages from the PyPi (https://pypi.org/) repository:
```
pip install -r requirements.txt
```

#### Working inside the environment

Whether or not using docker we manage the python dependencies with **conda**. This means that a python conda environment is found in the container, too. Inside the container, you can get into the environment like you normally would when using conda as package manager:

```
conda activate beak-ta3
```

You can run your code and tests from the command line. For example:

```
conda activate <env-name>
python <path/to/your/file.py>
```

To start with, open your IDE of choice and select the Python interpreter from the **beak-ta3** environment.

## 6. How to use
Because the repository behaves as a **local** python package 🐍, all functions can be accessed like so:

```python
# Import
from beak.module.submodule import function

# Call
function(arguments)
```

Even better, **data** 💾 can be accessed by:
```python
# Import
from importlib.resources import files
from 🦄 import load_function

# Path
file_path = str(files("beak.data") / "subfolder" / "file_name.ext")

# Load data
data = load_function(file_path)
```

## 5. For developers

### Information
If you are a developer, you might want to update the package or modify it.
Thanks to [MTRI](https://github.com/DARPA-CRITICALMAAS/mtri-packagetemplate), we can use the **repository code** like a **package**. To install everything from scratch, follow the instructions on the linked page, with some minor adjustments.

Since this repository provides an already initialized package (named *beak*), **you do not need to execute the inital** `pip install -e .` **after cloning the repository**. Only consider the steps below if you change something in the setup, e.g. version numbers or other entries ❗

### Install
There are some bottlenecks regarding the package requirements and channels they can installed from. For some of them, only pip works fine, for some, only conda, and others require previously installed packages and cannot installed with their prerequisites in one run. 

To solve this, we've chosen the way to set-up the environment as described using the
- **environments.yml** for the basic Conda setup and the
- **requirements.txt** for additional packages based on the [PyPi](https://pypi.org/) repository

To use the repository as a **local** Python package 🐍, you need to setup the environment correctly, either by:
1. install everything by your own and follow the [template](https://github.com/DARPA-CRITICALMAAS/mtri-packagetemplate) instructions or
2. **use one of the two described ways via Docker or Conda**

We recommend the **second way**, since it is approved and results in less headache 🤯.

🚨 The installation of all neccessary requirements by using the placeholders in the `setup.py` in the provided template will **not** work due to the explained dependency and availability issues for some packages.

Once the environment is ready: 
1. update pip: `pip install --upgrade pip`
2. install the wheel: `pip install wheel`
3. install the package: `pip install -e .`

The `-e` tells the environment that source files may change, so you don't have to re-run the `pip install` command after you edit your files. 

The package works with the provided repository, so you can simply use it as it is. If you decide for whatever reason to switch from **Docker** to **Conda** or vice-versa, you **do not** need to **re-run** the initialization steps either. It will work **just fine** 😎 as long as your're doing all your stuff in the repository folder 🥋. 
### Documentation
To build the documentation, the `sphinx` package is used. It currently works only in Linux/Docker. The provided template `builddocs.bash` may not work for everyone directly. The documentation files can be generated by running the `builddocs.bash` file from within the top-level folder of the package.

Here is how to solve common `command not found` or `encoding` issues (Windows, Linux and Mac systems have different file endings that may cause erros when executing scripts in the terminal.

**How to solve the `Command not found` error:**
1. Add the path with the `builddocs.bash` file to the `PATH` variables:
2. `export PATH=$PATH:/beak-ta3/`

**How to solve the `/bin/bash^M: bad interpreter: No such file or directory` error:**
1. Use a text editor like [Notepad++](https://notepad-plus-plus.org/)
2. Change the line ending to **Unix (LF)**
3. Save the file

Alternative:
1. Set the preferences for new documents to **Unix (LF)**
2. Copy/paste the file content
3. Save as another/overwrite the original file
