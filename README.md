# Beak-TA3
---
Advanced hybrid AI-methods for mineral predictive mapping

## Repository status

This repository is still in development. Documentation will grow over time.

Current contents include

- basic file structure
- environment and instructions

## For developers

### Prerequisites

All contributing developers need git, and a copy of the repository.

```
git clone <https://github.com/DARPA-CRITICALMAAS/beak-ta3.git>
```

Alternatively, you can also you GitHub Desktop instead of Git for repo-management.

After cloning, there are two options setting up your local development environment.

1. Docker
2. Conda

Docker is recommended as it containerizes the whole development environment, making sure it stays identical across different developers and operating systems. Using a container also keeps your own computer clean of all dependencies.

If you’re not familiar with development inside a container, see [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers) and [https://code.visualstudio.com/docs/devcontainers/tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) to prepare, e.g. for VSCODE

### Setting up a local development environment with docker (recommended)

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
conda activate dev-env
```

**Note** that your local repository gets automatically mounted into the container. This means that:

- The repository in your computer's filesystem and in the container are exactly the same
- Changes from either one carry over to the other instantly, without any need for restarting the container

For your workflow this means that:

- You can edit all files like you normally would (on your own computer, with your favourite text editor etc.)
- You must do all testing and running the code inside the container

### Python inside the container

Whether or not using docker we manage the python dependencies with **conda**. This means that a python conda environment is found in the container, too. Inside the container, you can get into the environment like you normally would when using conda as package manager:

```
conda activate dev-env
```

The conda environment for development is called **dev-env.** You’re welcome to create other environments for testing purpuses if you need to. To do so, either 
- Clone the existing environment or
- Create a new one using the dependencies.yml (most important packages defined) or the environment.yml (complete environment definition without build libs)

You can run your code and tests from the command line. For example:

```
conda activate <env-name>
python <path/to/your/file.py>
```

Or you can use Jupyter notebook or lab. However, depending on your IDE, you may have to install few extensions to be able to work with Jupyter. You also need to select your environment of choice, e.g. the **dev-env** as default for development.

### Set-up local conda environment

You can also set-up a local conda environment on your OS using the environment.yml from within the cloned repository folder:
```
conda env create -n <env-name> -f environment.yml
```
