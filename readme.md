# Physics Guided Path Planning Transformer

This repository contains the code for Physics Guided Path Planning Transformer (PGPPT)  - deep learning-based sequence-to-sequence model designed for efficient path planning of under-actuated autonomous marine vehicles in dynamic ocean environments. 


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#Notes)
- [Support and Contributions](#support-and-contributions)


## Introduction

This project contains the code for PGPPT.

## Installation
The project was run using Python 3.11, CUDA Version 12.2, torch 2.1.2

### Clone the project and install packages
```bash
# Clone the repository
git clone https://github.com/rohit1607/PGPPT.git

# Navigate to the project directory
cd PGPPT

# make a virutal environment and activate it
python -m venv my_venv
source my_venv/bin/activate

# Install dependencies
cd my-translat-transformer
pip install -r requirements.txt
```

### Modify ROOT and verify gym_examples installation

The following steps are required to complete the setup on your local machine.

Notes: 'path_to_your_dir' is the path of the directory where you cloned this repository

1. Navigate to the src folder

    ```bash
    # navigate to src folder
    cd path_to_your_dir/PGPPT/my-translat-transformer/src
    ```
2. Edit **root_path.py** to incorporate your 'path_to_your_dir' path.
    ```python
    ROOT = "path_to_your_dir/PGPPT/my-translat-transformer"
    ```
3. For simulating agents moving in a velocity field, we use custom environments implemented on top of OpenAI gym. To use them, verify if gym_examples is installed by running. 
    ```bash
     pip list | grep gym_examples
    ```
    If it is not installed, then do the following
    ```bash
    # navigate to gym-examples folder
    cd path_to_your_dir/PGPPT/my-translat-transformer/gym-examples

    # install gym_exmaples
    pip install -e .
    ```
4. Login to your wandb (weights and biases) account. See https://wandb.ai/site/ for details.

### Download data and place in the data folder
1. Download the data from here [---> DATA: ONEDRIVE_LINK <---](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/deepakns_iisc_ac_in/EgIYYl6AYY1EgDGMNbJz8GsB7FhlnnRXILBJ3y_JmSwg5Q?e=AtMf91) 

2. Move the downloaded files to the data folder and unzip them
    ```bash
    # navigate to the data folder
    cd path_to_your_dir/PGPPT/my-translat-transformer/data

    # move the downloaded files to the data folder and unzip them
    unzip DOLS_Cylinder.zip
    unzip GPT_dset_DG3.zip
    ```
2. After unzipping the files, contents of the data folder should be
    ```
    # path_to_your_dir/PGPPT/my-translat-transformer/data
    .
    ├── DOLS_Cylinder/
    ├── GPT_dset_DG3/
    ```

### Modify torch.nn.module.transformer to vizualize attention scores
To extract attention weights, torch.nn.modules.transformer was modified.
If you want to use visualize attention scores in this project, then you have to replace the library code with the one we have provided in src/customized_torch_nn_transformer.py
1. Open the following file:
    ```
    path_to_your_dir/PGPPT/my-translat-transformer/src/customized_torch_nn_transformer.py
    ```
2. Copy its contents. Ctrl+A, Ctrl+C
3. Open the torch.nn's transformer module by navigating to
    ```
    path_to_your_dir/PGPPT/my_venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py
    ```
    and delete its contents. Ctrl+A, Del
4. Paste the copied contents here. Crtl+V

Note:
- The modifications made to torch's transformer code can be seen by searching for the keyword EDIT in the customized version.

## Usage
1. To train the model
    ```bash
    # Navigate to src folder
    cd path_to_your_dir/PGPPT/my-translat-transformer/src

    # For Flow past cylindrical island scenario:
    python main.py --mode train --CFG v5_DOLS --quick_run False

    # For Double gyre:
    python main.py --mode train --CFG v5_GPT_DG3 --quick_run False
    ```
    
    Training config files for the two scenarios are located in the cfg folder.
    Experiment hyperparameters can be set there.
    ```
    path_to_your_dir/PGPPT/my-translat-transformer/cfg/contGrid_v5_DOLS.yaml

    path_to_your_dir/PGPPT/my-translat-transformer/cfg/contGrid_v5_GPT_DG3.yaml
    ```
    
    Notes:
    
    - Post training, inference on the test set will automatically be run on the best ckpt.
    - Each experiment is automatically named in the format "my_translat_{scenario}\_model\_{date_time}"
    - Results will be saved in the log folder:
        ```bash
        path_to_your_dir/PGPPT/my-translat-transformer/log
        ```
    - The log folder will contain 3 files for each experiment:
        - my_translat_{scenario}\_model\_{date_time}_src_stats.npy 
            - contains data statistics computed during pre-processing. Used for de-normalizing the data durign post-processing
        - my_translat_{scenario}\_model\_{date_time}.pt
            - contains the best model ckpt
        - my_translat_{scenario}\_model\_{date_time}.yml
            - is a copy of the cfg file that was used to run the experiment

2. For Running inference manually
    ```bash
    python main.py --mode inference_on_ckpt --ckpt ckpt_path_in_log
    ```
    Notes:
    - running in inference also populates the paper_plots directory 
    - plots showing trajectories in velocity fields will be populated inside the folder:
        ```
        paper_plots/exp_name/dataset_name/translation/
        ```
    - For DOLS_Cyliner dataset,
        - test_ip_op.png visualizes training data and test set predictions with arrival times
        - test_traj_ip_op.png visualizes the correctness of the path taken (North/South of island)
        - velocity.png visualizes the velocity fields
    
        
## Notes

1. gym-examples/gym_examples/envs contains custom implementations of simulated environments with an agent and velocity field

2. src/src_utils contains dataset implementations and othe project specific utility functions

3. Configuration files for experiments are present in the cfg folder
 
4. For each experiment, Model ckpts, experitment configuration, and data statisitics are logged in the log folder

5. Data should be downloaded, unzipped and kept inside the data folder

2. Users will require to create and login to their wandb account. This is required
for logging experiment metrics.


## Support and Contributions

If you encounter any issues or have questions, please feel free to [raise an issue](https://github.com/rohit1607/PGPPT/issues) on GitHub. Contributions, suggestions, and feedback are always welcome to help improve this project.
