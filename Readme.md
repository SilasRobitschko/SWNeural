# Neural density functional theory of liquid-liquid phase coexistence

This repository contains the code, datasets, and models accompanying the publication:

**Learning the physics of liquid-liquid phase separation**
> *To be published*

### Abstract
> We use simulation-based supervised machine learning and classical density functional theory to investigate bulk and interfacial phenomena associated with phase coexistence in binary mixtures. For a prototypical symmetrical Lennard-Jones mixture, our trained neural density functional yields accurate liquid-liquid and liquid-vapour binodals together with predictions for the variation of the associated interfacial tensions and contact angles across the entire fluid phase diagram. We investigate in detail the wetting behaviour at fluid-fluid interfaces along the line of triple-phase coexistence.

### Introduction

This codebase serves to demonstrate how a trained c1-Functional, within the framework of classical Density Functional Theory (cDFT), can be used to generate accurate density profiles for liquid-liquid and liquid-vapor transitions. \
Methods beyond profile generation, such as functional integration to calculate the free energy and surface tension, are outside the scope of this repository.

### Setup

Working in a virtual environment is highly recommended and make sure to have all the dependencies installed:

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate

# For Cuda support, consider running:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```
To use a GPU with TensorFlow, please refer to the official installation guide at [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

### Instructions

Key directories:
*   `scripts/`: Contains the scripts on how to use the trained Functionals to create interfacial profiles
*   `results/`: The default output directory for generated profiles.
*   `train/`: Contains the scripts for training a neural network 
*   `models/`: Contains pre-trained neural functional models.
*   `simdata/`: Contains the reference Grand Canonical Monte Carlo simulation data used for training.

#### Generating Density Profiles
The script `scripts/generate_profiles.py` demonstrates how to use a trained neural network to create self-consistent density profiles for various interfaces (liquid-liquid, vapor-demixed liquid, etc.).

To run with the pre-trained model, simply execute the script:
```bash
python3 scripts/generate_profiles.py
```
The resulting profiles will be saved in the `results/` directory.

#### Training a New Model
For users interested in training a new model from scratch, a sample script is provided in `train/learn.py`.

Before execution, it is advisable to configure training parameters (e.g., `epochs`, number of `workers`) by editing the `train/learn.py` script directly.

To start the training process, run:
```bash
python3 train/learn.py
```
**Note:** Training can be resource-intensive. For multi-process data loading, it may be necessary to increase the system's limit for open file descriptors (e.g., `ulimit -n 4096` on Linux/macOS). 

The network architecture can be found in `train/model.py` whereas the logic to generate the training dataset is located in `train/dataset.py`.


### Further Information

The reference data in `simdata/cluster_03_12_24` was generated with Grand Canonical Monte Carlo simulations using the [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd) software package.

The folder `train/mbd` and the script `train/runanalyzer.py` are used to process the reference data. 

The corresponding utility functions are available in `train/utils.py` and `scripts/utils.py`.