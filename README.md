# LoTTA: Low-rank Test-time Adaptation for Unsupervised Tabular Anomaly Detection

The code of paper "LoTTA: Low-rank Test-time Adaptation for Unsupervised Tabular Anomaly Detection"

This repository is not yet completed, so please check this as a reference only.
## Prepare dataset
   1) When using your own data, move the dataset into `./Data`. 
   2) Add the dataset name to `./Dataset/DataLoader.py` based on the format of your dataset.
   3) Modify *dataset_name* and *data_dim* in `./main.py`
   4) You can download tabular datasets from [ODDS](https://odds.cs.stonybrook.edu/) and [ADBench](https://github.com/Minqi824/ADBench) for testing.

## Run
Run `main.py` to start training and testing the model. Results will be automatically stored in `./output`.
