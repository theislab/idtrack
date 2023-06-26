# IDTrack Reproducibility Environments

Follow the steps.

1. Clone the repo.

    ```bash
    git clone https://github.com/theislab/idtrack
    ```

2. Necessary to run beforehand for MAC Silicon:

    > Note: Assumes that [homebrew](https://brew.sh) was already installed.

    ```bash
    brew install openblas
    brew install pyenv --head
    # Make sure package manager can find these paths. Export paths before the installation.
    export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
    export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/openblas/include"
    ```

3. Install the environment.

    > Note: [mamba](https://github.com/conda-forge/miniforge) (mambaforge) improves install times drastically.
    > All mamba commands can be replaced by `conda`.

    ```bash
    mamba env create -f environment/idt_env_minimal.yaml  # For M1 Mac. No GPU support.
    ```

    To remove the environment, run the following:

    ```bash
    mamba env remove -n idt_env
    mamba clean -avvvy
    ```

4. Create a branch branch `git checkout -b your_branch`

5. Build the jupter extensions.

    ```bash
    jupyter nbextension enable --py widgetsnbextension
    jupyter lab build
    ```
