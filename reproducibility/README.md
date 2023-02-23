# IDTrack Reproducibility

Environment guide.

1. Clone the repo.
    
    ```bash
    git clone https://github.com/theislab/idtrack
    ```

2. Install the environment, set up base mamba environment.
    
    `mamba` is strongly recommended over `conda`. 

    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

    ```

    ```bash
    mamba env create -f reproducibility/idt_env.yaml
    ```

3. Create a branch branch `git checkout -b your_branch`

4. Build the jupter extensions.

    ```bash
    jupyter nbextension enable --py widgetsnbextension
    jupyter lab build
    ```

5. Open `tutorial.ipynb` under `reproducibility` directory and make sure you can run all of the cells.