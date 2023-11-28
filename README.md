# Beak-TA3
Advanced hybrid AI-methods for mineral predictive mapping

## Repository status

This repository is still in development. Documentation will grow over time.

Last update: **11-28-2023**

What's new?
- slight repo structure modifications
- added **mvt_nat** experiment: modified data extraction from datacube and preparation for initial SOM model

Used tools are currently in the `/experiments/mvt_nat/utilities` module. These will be changed in future updates to match in a more intuitive module structure (e.g. tools for io, rasterization, ..)

Current contents include

- basic file structure
- environment and instructions

If you build the environment before 11-12-2023, please rebuild with the current updates. 

Only the **master branch** has up-to-date environment settings. <br>
If you set-up the environment, please **always** build from the main branch!

## Repository structure

The repository is build-up as following:

```
├── beak-ta3/
│   ├── docs/               # additional documentations
│   ├── experiments/        # mineral systems (e.g. MVT-NAT, MVT-REG, ...) working on
│   ├── local/              # stuff that's not beeing uploaded to GitHub
│   └── methods/            # Basic code for the MPM-methods in development
│       ├── bnn/
│       ├── hybrid/
│       ├── processing/
│       └── som/
│   └── utilities/          # helper functions   
```

The development for SOM, BNN and hybridization is accomplished on the respective branches:
- som_dev
- bnn_dev
- hybrid_dev

Depending on the development status, results will be merged to the main branch. <br>
Some may be created when starting the development.

## Prerequisites

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

## Setting up a local environment with docker

Create a CMD-instance from within the beak-ta3 folder. Build and run the container. Run this and every other command in the repository root unless otherwise directed.

```
docker compose up -d
```

If you need to rebuild already existing container (e.g. dependencies have been updated), run

```
docker compose up -d --build
```

### Working with the container

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

## Set-up local conda environment

You can also set-up a local conda environment on your OS using the environment.yml from within the cloned repository folder.

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

## Working inside the environment

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
