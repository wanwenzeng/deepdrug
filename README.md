![model](https://github.com/wanwenzeng/deepdrug/blob/main/model.png)

## Installation

* pytorch==1.8.1
* torch-geometric==1.7.1
* pytorch-lightning==1.3.6

## Usage

> python deepdrug.py --configfile ./config/DAVIS.regression.yml

parameters in the config file:

> dataset:  dataset name, such as "DAVIS"
> task: binary/multilabel/multiclass/regression
> category: DDI/DTI
> entry1_file/entry2_file: a file contain processed drug/protein graph features
> entry1_seq_file/entry2_seq_file:  a csv format file contains drug/protein sequence features
> pair_file: a file contains all sample (drug-drug or drug-protein) pairs
> label_file: a file contains labels corresponding to pair_file, which can be one-column integer (for binary classification task), multi-columns integers (for multi-class/label classification task) and one-column float numbers (for regression task).
> save_folder: a directory to save the outputs
> gpus: the gpu device ID to use.
> num_out_dim: 1 for binary classification and regression task, specific output dimension for multi-class/label classification task, such as 1317 for multi-label classification task for TwoSides dataset.
