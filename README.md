# DeepDDI
Drug-Drug/Protein interaction Prediction via a Graph Convolutional Network

[![BxSgR1.png](https://s3.ax1x.com/2020/11/12/BxSgR1.png)](https://imgchr.com/i/BxSgR1)

We use graph convolutional networks to learn the graphical representations of drugs and proteins such as molecular fingerprints and residual structures in order to boost the prediction accuracy. 
DeepDrug outperforms other state-of-the-art published methods both in terms of accuracy and robustness in predicting DDIs and DTIs with varying ratios of positive to negative training data


## Requirements
python=2.7.13
tensorflow=1.13.1=gpu_py27hcb41dfa_0
keras=2.1.6

for detail requirements check requirements.txt

## Installation
git clone https://github.com/wanwenzeng/DeepDrug

Installation has been tested in a Linux platform.

## Instructions
We prepared example data and detailed instructions to help you running our model. Feel free to contact us if you have further question.

Here we use Drug-Drug-interaction prediction for demonstration. Drug-Target(protein) prediciton shares same procedures.

### step 1: prepare data
The GCN model need adjacency matirx and corresponding feature data as its input, so our DDI project will have two pairs of adj_data and feat_data representing two drugs.

Here we use [DeepChem](https://github.com/deepchem/deepchem) library for extracting node features and gragh of a drug. The extraction code can be found in `./util/process_drugs.py`. 

The pre-extracted drug data can be found in ./data/drug_data.tar, which contains 10667 drug features.(So if you don't want to generate drug_data yourself, you should first unzip the tar file before running)

Following our paper you can track all the origin drug-drug relation data, here we only provide a preprocessed relation data from drugbank, which you can find in ./data/idxDrugBank-01.csv

Before further running, check all paths and make sure you renamed with your local env paths.
### step 2: seperate train/test data

After all the preparation work are done, we are ready to split&save the data for further training.

run
`python main.py -type run -max_atom 50 -sample 1_1 -gpu_id 1 -israndom 1 -database drugbank`

-max_atom  the max graph nodes we will have in the GCN model.
-sample    the ratio of positive data and negative data
-gpu_id    gpu id
-israndom  rather padding the small drug(smaller than max_atom) with random values or zeros.
-database  which database to use(drugbank, twosides, etc)

After running this part code, you will have `train_data.npz`, `test_data.npz` saved for further training.

### step 3: train and test
Finally we can train our model and tested it.
run

`python main.py -type train -sample 1_1 -max_atoms 50 -israndom 1 -database drugbank -location test -unit_list 128 128 128 128 128 128 128 128 -use_bn 1 -use_relu 1 -use_GMP 1 -gpu_id 1`

-location  the test result(auc, auprc, pics, etc) save path
-unit_list unit list for graph layers
-use_bn    use batch normalization for GCN
-use_relu  use relu as the activation method
-use_GMP   use globalMaxPooling for GCN

This will give you the test result of GCN, lr, and rfc.

## Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact us 
[caoxusheng@mail.nankai.edu.cn](caoxusheng@mail.nankai.edu.cn)
[froxanne1024@gmail.com](froxanne1024@gmail.com)
