#coding=UTF-8
from util import getDrugFeatures
import csv
from keras.models import load_model
from layers.graph import GraphLayer, GraphConv
import pickle
import numpy as np
from sklearn.externals import joblib

def getSpecialX():
    drug1 = getDrugFeatures('DB00563', 40, 1)
    drug2 = getDrugFeatures('DB01115', 40, 1)
    drug_feat1 = drug1[0]
    drug_adj1 = drug1[1]
    drug_feat2 = drug2[0]
    drug_adj2 = drug2[1]
    drug_feat1 = drug_feat1.reshape(-1, 40, 75)
    drug_adj1 = drug_adj1.reshape(-1, 40, 40)
    drug_feat2 = drug_feat2.reshape(-1, 40, 75)
    drug_adj2 = drug_adj2.reshape(-1, 40, 40)
    return [drug_feat1, drug_adj1,drug_feat2, drug_adj2]


def loadModels():
    gcn_model = load_model('/home/zengwanwen/caoxusheng/DeepDDI_desktop/checkpoint/MyBestDeepCDR__2013_stanford_twosides_1_8_2020-08-28 13:16:51.392238.h5',\
        custom_objects={'GraphConv': GraphConv})
    lr_model = joblib.load('/home/zengwanwen/caoxusheng/DeepDDI_desktop/checkpoint/lr/2013_stanford_train_twosides_test_1_8.pkl')
    # with open('/home/zengwanwen/caoxusheng/DeepDDI_desktop/checkpoint/lr/twosides_stanford_train_2013_test_1_1.pkl', 'rb') as lr:
    #     lr_model = pickle.load(lr)
    # with open('./trained_models/rfc.pickle', 'rb') as frfc:
    #     rfc_model = pickle.load(frfc)
    return gcn_model, lr_model#, rfc_model

def demo(Max_atoms, israndom):
    gcn_model = loadModels()
    while True:
        print("please type two drugs you want to tested")
        drug_id1 = raw_input("the first drug:  ")
        drug1 = getDrugFeatures(drug_id1, Max_atoms, israndom)
        drug1 is None
        if drug1 is None:
            print("haven't taken this drug in consideration.")
            continue

        drug_id2 = raw_input("the second drug: ")
        drug2 = getDrugFeatures(drug_id2, Max_atoms, israndom)
        if drug2 is None:
            print("haven't taken this drug in consideration.")
            continue
        drug_feat1 = drug1[0]
        drug_adj1 = drug1[1]
        drug_feat2 = drug2[0]
        drug_adj2 = drug2[1]
        drug_feat1 = drug_feat1.reshape(1, Max_atoms, 75)
        drug_adj1 = drug_adj1.reshape(1, Max_atoms, Max_atoms)
        drug_feat2 = drug_feat2.reshape(1, Max_atoms, 75)
        drug_adj2 = drug_adj2.reshape(1, Max_atoms, Max_atoms)

        result = gcn_model.predict([drug_feat1, drug_adj1, drug_feat2, drug_adj2])
        print("the gcn predicted result is: ", result[0][0])


def predict(model, lr, drug_id1, drug_id2, Max_atoms, israndom):
    drug1 = getDrugFeatures(drug_id1, Max_atoms, israndom)
    drug2 = getDrugFeatures(drug_id2, Max_atoms, israndom)
    drug_feat1 = drug1[0]
    drug_adj1 = drug1[1]
    drug_feat2 = drug2[0]
    drug_adj2 = drug2[1]
    drug_feat1 = drug_feat1.reshape(-1, Max_atoms, 75)
    drug_adj1 = drug_adj1.reshape(-1, Max_atoms, Max_atoms)
    drug_feat2 = drug_feat2.reshape(-1, Max_atoms, 75)
    drug_adj2 = drug_adj2.reshape(-1, Max_atoms, Max_atoms)
    x = np.c_[drug_feat1.reshape(1, 75*Max_atoms), drug_adj1.reshape(1, Max_atoms*Max_atoms),
              drug_feat2.reshape(1, 75*Max_atoms), drug_adj2.reshape(1, Max_atoms*Max_atoms)]
    lr_result = lr.predict(x)
    result = model.predict([drug_feat1, drug_adj1,drug_feat2, \
        drug_adj2])
    return result[0][0], lr_result


def multiDemo(database, sample, Max_atoms, israndom):
    read_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/' + \
            database + '/' + sample + '/' + 'index.csv'
    in_fine = open(read_path, 'r')
    idx_reader = csv.reader(in_fine)
    gcn_model, lr = loadModels()
    i = 0
    for item in idx_reader:
        result = predict(gcn_model, lr, item[0], item[1],\
             Max_atoms, israndom)
        i += 1
        print(" ")
        print("demo ", i, "is below:")
        print("drug1: ", item[0], "  drug2: ", item[1])
        print("the real result is:       ", item[2])
        print("the gcn predicted result is:  ", result[0])
        print("the lr predicted result is:  ", result[1][0])
        print(" ")



