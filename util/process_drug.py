#get drug features using Deepchem library
import os
import deepchem as dc
from rdkit import Chem
import hickle as hkl
import csv
import argparse
drug_smiles_file='../data/structure links.csv'
save_dir='../data/GDSC/drug_graph_feat_new2'



parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, help='GPU devices')
args = parser.parse_args()

reader = csv.reader(open(drug_smiles_file,'r'))
rows = [item for item in reader]
rows = rows[1:len(rows)]
pubchemid2smile = {item[0]:item[6] for item in rows if item[5]!=''}

# for each in pubchemid2smile.values():
#     print(each)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id



if not os.path.exists(save_dir):
    os.makedirs(save_dir)
molecules = []
for each in pubchemid2smile.keys():
    print(each)
    molecules=[]
    molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    try:
        mol_object = featurizer.featurize(mols=molecules)
    except ValueError as e:
        continue
    try:
        features = mol_object[0].atom_features
    except AttributeError as e:
        continue
    # try:
    #     features = mol_object[0].atom_features
    # except AttributeError as e:
    # #error: has not attribute
    #     pass
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list
    hkl.dump([features,adj_list,degree_list],'%s/%s.hkl'%(save_dir,each))
    feat_mat,adj_list,degree_list = hkl.load('%s/%s.hkl'%(save_dir,each))
    print(feat_mat)



