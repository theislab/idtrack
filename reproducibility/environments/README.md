# Development Guide for IDTrack Package

This detailed guide will assist you in setting up your development environment for working with IDTrack, ensuring a smooth and reproducible workflow.

### 1. Cloning the Repository

Start by cloning the IDTrack repository from GitHub. This will download the latest version of the source code to your local machine:

```bash
git clone https://github.com/theislab/idtrack
```

### 2. Pre-Installation Setup for macOS Silicon Users

Before proceeding with the installation on macOS Silicon, you need to ensure that certain dependencies are installed. This step is crucial for ensuring that the environment runs smoothly on your hardware.

-   **Install Homebrew**: First, verify that you have Homebrew installed, as it will be used to install other necessary packages. You can install or check for Homebrew by following the instructions on [Homebrew's official site](https://brew.sh).

-   **Install Dependencies**:

    -   Install the hdf5:
        ```bash
        brew install hdf5
        ```
    -   Install Pandoc for handling document conversions for docs built:
        `bash brew install pandoc `
    -   Set environment variables to ensure these libraries are correctly recognized by your package manager:
        ```bash
        export BREW_PREFIX="$(brew --prefix)"
        export HDF5_DIR="${BREW_PREFIX}/opt/hdf5"
        export HDF5_INCLUDE="${HDF5_DIR}/include"
        export HDF5_LIB="${HDFWIFT_DIR}/lib"
        ```

### 3. Setting Up Conda Development Environments

For managing Python packages and ensuring that your development environment is isolated and reproducible, use Conda or Mamba. Mamba is a faster alternative to Conda and can significantly speed up package installations.

-   **Create the Environment**:

    Creating isolated Conda environments is essential for maintaining project consistency and ensuring that dependencies do not interfere with one another. This section will guide you through setting up two distinct Conda environments using Mamba, a faster alternative to Conda. One environment is tailored for managing Poetry dependencies and executing related commands, while the other is designed for general development and testing purposes.

    1.  Creating the Poetry Environment: To optimize your workflow for handling Poetry dependencies.

        -   **Install Mambaforge**: If Mamba is not yet installed, download it from [the Conda Forge page](https://github.com/conda-forge/miniforge) to enhance installation speeds and manage environments more efficiently.
        -   **Create the Environment**: Utilize the YAML file provided in the repository to set up the environment specific to Poetry. **This environment is necessary for running the commands detailed in section 4 of this guide**.

        ```bash
        mamba env create -f reproducibility/environments/idtrack_poetry_3_9_env.yaml
        ```

        -   Version Compatibility: Be aware that future updates are planned to extend support to Python versions 3.10 through 3.12, ensuring compatibility with newer Python releases.

    2.  Creating the Development and Testing Environment: For broader development and testing activities.
        -   **Create the Environment**: Use another YAML file from the repository to establish an environment suited for the main development and testing tasks.
        ```bash
        mamba env create -f reproducibility/environments/idtrack_dev_env.yaml
        ```

    By carefully managing these environments, you can effectively segregate different types of activities (such as development, testing, and dependency management) within your project, leading to a cleaner and more efficient development process.

-   **Activating and Using the Environment**:

    -   To activate the newly created environment, use:
        ```bash
        conda activate idtrack_poetry_3_9_env
        ```
    -   You are now ready to run IDTrack within this isolated environment.
    -   Run the following to make sure the widgets etc is built:
        ```bash
        jupyter lab build
        ```

-   **Removing the Environment**:

    -   If you need to remove the environment for any reason, you can do so cleanly with the following commands:
        ```bash
        mamba env remove -n idtrack_poetry_3_9_env
        mamba clean -avvvy
        ```

-   **Apple Silicon Specific Steps**:
    -   Ensure you have the HDF5 library installed, as it is crucial for certain data handling operations within IDTrack. This library is included in the Conda environments to facilitate smooth operation on Apple hardware.

By following this guide, you should have a robust, reproducible environment ready for developing and testing with IDTrack. This setup not only ensures compatibility across different systems but also enhances the overall efficiency and reliability of your development workflow.

### 4. Install the poetry environments.

Poetry is a tool for Python package management that simplifies declaring, managing, and installing project dependencies. It ensures that package installations are reproducible by using a lock file to pin specific versions of dependencies. Integrating Poetry within a Conda environment for Python package development provides a streamlined workflow, where Poetry handles dependency management and Conda manages the project environment. This setup enhances project consistency and minimizes compatibility issues across different development stages. For more details on managing environments with Poetry, explore this website.

-   **Updating the Poetry Lock File and Installing Dependencies:**

    When you modify your poetry configuration, it is essential to update the lock file to ensure all dependencies are compatible and then reinstall the environment. The command starts by activating the Conda environment specifically set for the poetry project. It then navigates to the project directory. The `poetry lock` command updates the lock file, which ensures all dependencies are recorded correctly. Following this, `poetry install` installs or updates the dependencies based on the new lock file. The environment is then deactivated with `conda deactivate`, and finally, the command returns to the home directory.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack; poetry lock; poetry install; conda deactivate; cd
    ```

-   **Running Pre-commit Hooks:**

    When running pre-commit hooks to check code quality and standards, the below command is used. It activates the Conda environment where the poetry project resides and navigates to the project directory. The `poetry run pre-commit run --all-files` command runs the pre-commit hooks against all files in the repository, which helps in identifying and fixing issues before they are committed to the version control system. After running the hooks, the Conda environment is deactivated, and the command exits to the home directory.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack; poetry run pre-commit run --all-files; conda deactivate; cd
    ```

-   **Performing Static Type Checking with Mypy:**

    To perform static type checking across the Python project, the command provided is tailored to ensure the source code complies with type annotations. Initially, it activates the Conda environment specifically configured for this project. The command then navigates to the project directory and employs poetry run mypy idtrack tests docs/conf.py to execute the mypy tool. mypy is a static type checker for Python, used to analyze the code in the specified directories—idtrack, tests, and the docs/conf.py configuration file. This helps detect type errors that can prevent runtime issues, ensuring the code adheres to declared types. After completing the type checking, the environment is deactivated, and the command returns to the home directory, maintaining a clean and organized workspace.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack; poetry run mypy idtrack tests docs/conf.py; conda deactivate; cd
    ```

-   **Generating and Viewing HTML Documentation:**

    For generating and viewing the HTML documentation of the project, the following command sequence is employed. It activates the appropriate Conda environment and changes to the documentation directory of the project. The `rm -rf _build` command removes the existing build directory to start fresh, preventing any stale files from being included. The `poetry run make html` command generates new HTML files for the documentation. The generated HTML files are then opened in the default web browser using `open _build/html/index.html`. Finally, the environment is deactivated, and the command navigates back to the home directory.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack/docs; rm -rf _build; poetry run make html; open _build/html/index.html; conda deactivate; cd
    ```

-   **Running Tests:**

    To execute the project's test suite and verify that all components are functioning as expected, this command is utilized. After activating the Conda environment, it navigates to the project's main directory. The `poetry run pytest --typeguard-packages=idtrack` command runs the pytest framework with type checking on the specified packages, ensuring that type annotations are used correctly throughout the project. After the tests are complete, the environment is deactivated, and the command line returns to the home directory.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack; poetry run pytest --typeguard-packages=idtrack; conda deactivate; cd
    ```

-   **Conducting a Security Vulnerability Check:**

    For ensuring the security of the project's dependencies, the following command is run. It starts by activating the Conda environment designated for the project and navigating to the project directory. The `safety check --full-report --file=.nox/safety-3-9/tmp/requirements.txt` command performs a security vulnerability check against the dependencies listed in the specified requirements file, providing a full report of any issues found. Following the security check, the environment is deactivated, and the command exits to the home directory.

    ```bash
    conda activate idtrack_poetry_3_9_env; cd /Users/kemalinecik/git_nosync/idtrack; safety check --full-report --file=.nox/safety-3-9/tmp/requirements.txt; conda deactivate; cd
    ```
