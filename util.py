import random
import csv, os, sys
import hickle as hkl
import numpy as np
import scipy.sparse as sp

DPATH = '/home/zengwanwen/caoxusheng/DeepDDI/data'
Drug_feature_file = '%s/GDSC/drug_graph_feat_new' % DPATH

data_idx_stanford = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/dataidx_stanford.csv'
data_idx_two_sides = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/dataidx_twosides_new.csv'
data_idx_decagon = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/dataidx_decagon.csv'
data_idx_2013 = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/idx2013.csv'
data_idx_2013_2c = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/idx2013-01.csv'
data_idx_drugbank_2c = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/idxDrugBank-01.csv'
data_idx_drugbank_3c = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/idxDrugBank.csv'

def DataSplit(dataX, dataY, ratio=0.90):
    random.seed(0)
    size = dataX.shape[0]
    slice_whole = range(size)
    slice_train = random.sample(slice_whole, int(ratio * size))
    slice_val = [i for i in slice_whole if i not in slice_train]
    X_train = dataX[slice_train, :]
    X_val = dataX[slice_val, :]
    y_train = dataY[slice_train]
    y_val = dataY[slice_val]
    return X_train, X_val, y_train, y_val


def DataSplitForRun(data_idx, ratio=0.95):  # leave drug out
    print('Begin to split data')
    data_train_idx, data_test_idx = [], []
    data_train = random.sample(data_idx, int(ratio * len(data_idx)))
    print('Finish train set')
    print(len(data_idx))
    print(len(data_train))
    data_test = [item for item in data_idx if item not in data_train]
    print('Finish split data')
    return data_train, data_test


def MetadataGenerate(database, sample):
    # 2850 pos 69160 neg
    if database == 'decagon':
        data_idx = csv.reader(open(data_idx_decagon, 'r'))
    elif database == 'twosides':
        data_idx = csv.reader(open(data_idx_two_sides, 'r'))
    elif database == 'stanford':
        data_idx = csv.reader(open(data_idx_stanford, 'r'))
    elif database == '2013':
        data_idx = csv.reader(open(data_idx_2013_2c,'r'))
    elif database == 'drugbank':
        data_idx = csv.reader(open(data_idx_drugbank_2c,'r'))
    pos_data_idx = []
    neg_data_idx = []
    data_idx = [tuple(item) for item in data_idx]

    for item in data_idx:
        if item[2] == '1':
            pos_data_idx.append(item)
        else:
            neg_data_idx.append(item)

    if sample == '1_1':
        if pos_data_idx<=neg_data_idx:
            size = len(pos_data_idx)
            neg_data_idx = neg_data_idx[:size]
        else:
            size = len(neg_data_idx)
            pos_data_idx = pos_data_idx[:size]
    if sample == '1_2':
        pos_size = len(pos_data_idx)
        neg_size = len(neg_data_idx)
        if 2*pos_size<=neg_size:
            neg_data_idx = neg_data_idx[:2*pos_size]
        else:
            pos_data_idx = pos_data_idx[:neg_size/2]
    if sample == '1_4':
        pos_size = len(pos_data_idx)
        neg_size = len(neg_data_idx)
        if 4*pos_size<=neg_size:
            neg_data_idx = neg_data_idx[:4*pos_size]
        else:
            pos_data_idx = pos_data_idx[:neg_size/4]
    if sample == '1_8':
        pos_size = len(pos_data_idx)
        neg_size = len(neg_data_idx)
        if 8*pos_size<=neg_size:
            neg_data_idx = neg_data_idx[:8*pos_size]
        else:
            pos_data_idx = pos_data_idx[:neg_size/8]
    if sample == '1_16':
        pos_size = len(pos_data_idx)
        neg_size = len(neg_data_idx)
        if 16*pos_size<=neg_size:
            neg_data_idx = neg_data_idx[:16*pos_size]
        else:
            pos_data_idx = pos_data_idx[:neg_size/16]

    print('MetadataGenerate  neg:',len(neg_data_idx))
    print('MetadataGenerate  pos:',len(pos_data_idx))
    neg_data_idx.extend(pos_data_idx)
    random.shuffle(neg_data_idx)
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        print each.split('.')[0]
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())
    print('Done metadata!')
    return drug_feature, neg_data_idx

    # could speed up using multiprocess and map
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix


def CalculateGraphFeat(feat_mat, adj_list, Max_atoms, israndom):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms, feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms - feat_mat.shape[0])
    feat[:feat_mat.shape[0], :] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    adj_ = adj_mat[:len(adj_list), :len(adj_list)]
    adj_2 = adj_mat[len(adj_list):, len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
    return [feat, adj_mat]

def FeatureExtract(data_idx, drug_feature, Max_atoms, israndom):
    print('Begin Feature Extracting...')
    nb_instance = len(data_idx)
    drug_data1 = []
    drug_data2 = []

    target = np.zeros(nb_instance, dtype='float32') #TODO
    idx_num = 0
    number_of_ignore=0

    for idx in range(nb_instance):
        print(idx)
        pubchem_id1, pubchem_id2, result = data_idx[idx]
        # modify
        feat_mat1, adj_list1, _ = drug_feature[str(pubchem_id1)]
        feat_mat2, adj_list2, _ = drug_feature[str(pubchem_id2)]
        # print(feat_mat1.shape, feat_mat2.shape)
        if (int(feat_mat1.shape[0]) > Max_atoms or int(feat_mat2.shape[0]) > Max_atoms):
            # print("too big")
            # print(feat_mat1.shape, feat_mat2.shape)
            number_of_ignore=number_of_ignore+1
            continue
        target[idx_num] = result
        drug_data1.append(CalculateGraphFeat(feat_mat1, adj_list1, Max_atoms, israndom))
        # if(idx==1):
        #     print('adj_list1')
        #     print(adj_list1)
        drug_data2.append(CalculateGraphFeat(feat_mat2, adj_list2, Max_atoms, israndom))
        idx_num += 1
    print("number_of_ignore:")
    print(number_of_ignore)
    target = target[:idx_num]
    print('Finish Feature Extracting')
    return drug_data1, drug_data2, target


def getDrugFeatures(drug_id, Max_atoms, israndom):
    flag = False
    for each in os.listdir(Drug_feature_file):
        if each.split('.')[0] == drug_id:
            flag = True
            feat_mat, adj_list, degree_list = hkl.load\
                ('%s/%s' % (Drug_feature_file, each))
    if flag:
        return CalculateGraphFeat(feat_mat, adj_list, Max_atoms, israndom)
    else:
        return None