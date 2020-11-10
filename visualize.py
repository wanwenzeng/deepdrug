import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser(description='Drug_Drug_Interaction main function')
parser.add_argument('-method', dest='method', type=str, help='tsne or pca')
parser.add_argument('-database', dest='database', type=str, help="use which database, stanford, drugbank, twosides, 2013")

args = parser.parse_args()

data = np.load('/home/zengwanwen/caoxusheng/DeepDDI_desktop/prog/low_dim_data/'+ args.database +'_'+ args.method +'.npz')
drug = data['drug']
fea = data['fea']
fea1 = fea[:,0]
fea2 = fea[:,1]

if args.database == 'drugbank':
    DDI = csv.reader(open('/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/drugbank/1_1/2c/index.csv', 'r'))
else:
    DDI = csv.reader(open('/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/'+args.database+'/1_1/index.csv', 'r'))

def scatter():
    ddi = dict()   
    for item in DDI:
        if item[2]=='1':
            if item[0] not in ddi:
                ddi[item[0]]=[]
            ddi[item[0]].append(item[1])

    for i in ddi:
        plt.cla()
        plt.scatter(fea1, fea2, c='y', alpha=0.5)
        for j in ddi[i]:
            index = np.argwhere(drug == j)
            plt.scatter(fea1[index], fea2[index],c='k')
        index = np.argwhere(drug == i)
        plt.scatter(fea1[index], fea2[index],c='b',s=40, marker='X')
        plt.savefig('/home/zengwanwen/caoxusheng/DeepDDI_desktop/prog/low_dim_data/result-' + args.method + '/'+args.database + '/'+ i +'.png')

scatter()