#coding=UTF-8
from util import DataSplitForRun, MetadataGenerate
from util import FeatureExtract
import gc
import numpy as np
import random
import csv
DPATH = '/home/zengwanwen/caoxusheng/DeepDDI/data'
Drug_feature_file = '%s/GDSC/drug_graph_feat_new' % DPATH

def SaveTrainData(X_drug_data_train1, X_drug_data_train2, Y_train, sample, database, Max_atoms):
    X_drug_feat_data_train1 = [item[0] for item in X_drug_data_train1]
    X_drug_adj_data_train1 = [item[1] for item in X_drug_data_train1]
    X_drug_feat_data_train2 = [item[0] for item in X_drug_data_train2]
    X_drug_adj_data_train2 = [item[1] for item in X_drug_data_train2]

    print('feat1',len(X_drug_feat_data_train1))
    X_drug_feat_data_train1 = np.array(X_drug_feat_data_train1)  # nb_instance * Max_stom * feat_dim
    print('adj1',len(X_drug_adj_data_train1))
    X_drug_adj_data_train1 = np.array(X_drug_adj_data_train1)  # nb_instance * Max_stom * Max_stom
    print('feat2',len(X_drug_feat_data_train2))
    X_drug_feat_data_train2 = np.array(X_drug_feat_data_train2)  # nb_instance * Max_stom * feat_dim
    print('adj2',len(X_drug_adj_data_train2))
    X_drug_adj_data_train2 = np.array(X_drug_adj_data_train2)  # nb_instance * Max_stom * Max_stom MEMORY ERROR    

    size = X_drug_adj_data_train2.shape[0]
    x = np.c_[X_drug_feat_data_train1.reshape(size, 75*Max_atoms), X_drug_adj_data_train1.reshape(size,
                            Max_atoms*Max_atoms), X_drug_feat_data_train2.reshape(size, 75*Max_atoms), X_drug_adj_data_train2.reshape(size, Max_atoms*Max_atoms)]
    # 保存到指定文件夹
    print('Begin to save train')
    save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/' + database + '/' + sample + '/4c'
    np.savez('%s/train_data.npz' % (save_path),X_train=x, y_train=Y_train)


def SaveTestData(X_drug_data_test1, X_drug_data_test2, Y_test, sample, database, Max_atoms):
    X_drug_feat_data_test1 = [item[0] for item in X_drug_data_test1]
    X_drug_adj_data_test1 = [item[1] for item in X_drug_data_test1]
    X_drug_feat_data_test2 = [item[0] for item in X_drug_data_test2]
    X_drug_adj_data_test2 = [item[1] for item in X_drug_data_test2]

    X_drug_feat_data_test1 = np.array(X_drug_feat_data_test1)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test1 = np.array(X_drug_adj_data_test1)  # nb_instance * Max_stom * Max_stom
    X_drug_feat_data_test2 = np.array(X_drug_feat_data_test2)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test2 = np.array(X_drug_adj_data_test2)  # nb_instance * Max_stom * Max_stom

    x_test = np.c_[X_drug_feat_data_test1.reshape(-1, 75*Max_atoms), 
                    X_drug_adj_data_test1.reshape(-1,Max_atoms*Max_atoms), 
                    X_drug_feat_data_test2.reshape(-1, 75*Max_atoms), 
                    X_drug_adj_data_test2.reshape(-1, Max_atoms*Max_atoms)]

    print('Begin to save test')
    save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/' + database + '/' + sample + '/4c/'
    np.savez('%s/test_data.npz' % (save_path), X_test=x_test, y_test=Y_test)


def run(max_atoms, database, sample, israndom):
    random.seed(0)
    # Meta data loading
    drug_feature, data_idx = MetadataGenerate(database, sample)
    save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/' \
        + database + '/' + sample + '/4c/' + 'index.csv'
    idx_file = open(save_path, 'w')
    idx_writer = csv.writer(idx_file)
    idx_writer.writerows([item for item in data_idx])
    idx_file.close()
    data_train_idx, data_test_idx = DataSplitForRun(data_idx)

    # Extract features for training and test
    X_drug_data_train1, X_drug_data_train2, Y_train = FeatureExtract(data_train_idx, drug_feature, max_atoms, israndom)
    SaveTrainData(X_drug_data_train1, X_drug_data_train2, Y_train, sample, database, max_atoms)
    train_len=len(X_drug_data_train1)
    del X_drug_data_train1
    del X_drug_data_train2
    del Y_train
    gc.collect()
    # print(data_test_idx)
    X_drug_data_test1, X_drug_data_test2, Y_test = FeatureExtract(data_test_idx, drug_feature, max_atoms, israndom)
    SaveTestData(X_drug_data_test1, X_drug_data_test2, Y_test, sample, database, max_atoms)
    test_len=len(X_drug_data_test1)
    del X_drug_data_test1
    del X_drug_data_test2
    del Y_test
    gc.collect()
    # print('Evaluation finished!')
    save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/' + database + '/' + sample + '/'
    print('文件存储位置:%s'% (save_path))
    print('Max_atoms:', max_atoms)
    print('train size:',train_len)
    print('test size:',test_len)