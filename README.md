# Active Learning Study

This is a simple study on different active learning method on the MNIST dataset implemented in Pytorch Lightning.

## Setup

Download the MNIST data set from [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv] and save the .csv files in a directory. The default directory read by the project is `MNIST`.

Use Python package manager of choice to install the package in `requirements.txt`.

## Running the experiments

Experiments are defined as a .json file. In `sample_setup.json` the basic configuration is shown. Run the experiments by the command `python main.py --setup-file sample_setup.json --data-dir MNIST`.
