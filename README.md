![model](https://github.com/wanwenzeng/deepdrug/blob/main/model.png)
<h3 align="center">
<p> DeepDrug: A general graph-based deep learning framework  <br>for drug-drug interactions and drug-target interactions prediction<br></h3>
---
DeepDrug is a deep learning framework, using residual graph convolutional networks (RGCNs) and convolutional networks (CNNs) to learn the comprehensive structural and sequential representations of drugs and proteins in order to boost the drug-drug interactions(DDIs) and drug-target interactions(DTIs) prediction accuracy. 

## Installation

* pytorch==1.8.1
* torch-geometric==1.7.1
* pytorch-lightning==1.3.6

## Usage

> python deepdrug.py --configfile ./config/KIBA.regression.yml

parameters in the config file:

> dataset:  dataset name, such as "KIBA"  
> task: binary/multilabel/multiclass/regression  
> category: DDI/DTI  
> entry1_file/entry2_file: a file contain processed drug/protein graph features  
> entry1_seq_file/entry2_seq_file:  a csv format file contains drug/protein sequence features  
> pair_file: a file contains all sample (drug-drug or drug-protein) pairs  
> label_file: a file contains labels corresponding to pair_file, which can be one-column integer (for binary classification task), multi-columns  integers (for multi-class/label classification task) and one-column float numbers (for regression task).  
> save_folder: a directory to save the outputs  
> gpus: the gpu device ID to use.  
> num_out_dim: 1 for binary classification and regression task, specific output dimension for multi-class/label classification task, such as 1317 for multi-label classification task for TwoSides dataset.  


## Data Processing 

For drugs, the structural features can be easily constructed by 
```python 
from dataset import EntryDataset
drug_df = pd.read_csv('drug.csv') 
save_folder = '/path/to/drug/graph/'
dataset = EntryDataset(save_folder)
dataset.drug_process(drug_df)
# graph features will be processed and saved in the /path/to/drug/graph/processed/data.pt 
```

For proteins, the structural features are constructed by [PAIRPred](https://onlinelibrary.wiley.com/doi/10.1002/prot.24479) software (processed feature data will be released soon).
```python
from dataset import EntryDataset
target_df = pd.read_csv('target.csv') 
target_graph_dict = np.load('PAIRPred_feature_dict.npz')[()]
save_folder = '/path/to/target/graph/'
dataset = EntryDataset(save_folder)
dataset.protein_process(target_df,target_graph_dict)
# graph features will be processed and saved in the /path/to/target/graph/processed/data.pt 
```

## Cite Us
If you found this package useful, please cite [our paper](https://www.biorxiv.org/content/10.1101/2020.11.09.375626v2):
```
@article{DeepDrug,
  title={DeepDrug: A general graph-based deep learning framework for drug-drug interactions and drug-target interactions prediction},
  author={Yin, Qijin and Cao, Xusheng and Fan, Rui and Liu, Qiao and Jiang, Rui and Zeng, Wanwen},
  journal={bioRxiv},
  year={2022}
}
```
