import random
from util import DataSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score,classification_report
import numpy as np
from lr_model import KerasMultiSourceGCNModel
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, History
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import os
import pickle
import matplotlib.pyplot as plt
from demo import getSpecialX

def set_data(database, sample, unit_list, use_bn, use_relu, use_GMP, location):
    result_save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/results/'+ \
        database + '/' + sample+'/'+ location+'/'
    out_path='%sout.txt' % result_save_path
    doc = open(out_path,'w+')
    print >> doc,'result save path:',result_save_path
    print >> doc,'unit_list:',unit_list
    print >> doc,'use_bn:',use_bn
    print >> doc,'use_relu:',use_relu
    print >> doc,'use_GMP:',use_GMP
    print >> doc,'database:',database
    print >> doc,'sample:',sample
    return doc


class MyCallback(Callback):
    def __init__(self, sample, validation_data, patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.sample = sample
        self.patience = patience
    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('./trained_models/deep_ddi_%s_%s.h5' % (self.sample, str(datetime.datetime.now())))
        print("the real y is:       ", self.y_val)
        print("the predicted y is:  ", self.model.predict(self.x_val))
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        # the_test = self.model.predict(getSpecialX())
        # print("the test is ", the_test[0][0])
        # print("the real y is:       ", self.y_val)
        # print("the predicted y is:  ", y_pred_val)
        auc_score = roc_auc_score(self.y_val, y_pred_val)
        print('the auc score is : %s' % auc_score)
        if auc_score > self.best:
            self.best = auc_score
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return


def ModelTraining(Max_atoms, doc, sample, lrModel, rfc, GCNModel, X_drug_feat_data_train1, X_drug_adj_data_train1, X_drug_feat_data_train2,
                  X_drug_adj_data_train2, Y_train, validation_data, nb_epoch=10):
    random.seed(0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    GCNModel.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    # EarlyStopping(monitor='val_loss',patience=5)
    callbacks = [ModelCheckpoint('./trained_models/deep_ddi_%s.h5' % sample, monitor='val_loss', save_best_only=False,save_weights_only=False),MyCallback(sample, validation_data=validation_data, patience=10)]
    size = X_drug_adj_data_train2.shape[0]
    x = np.c_[X_drug_feat_data_train1.reshape(size, 75*Max_atoms), X_drug_adj_data_train1.reshape(size, Max_atoms*Max_atoms),X_drug_feat_data_train2.reshape(size, 75*Max_atoms), X_drug_adj_data_train2.reshape(size, Max_atoms*Max_atoms)]
    print("*************this is x*****************")
    # print(x)
    lr_time=time.time()
    lrModel.fit(x, Y_train)
    lr_time=time.time()-lr_time
    print('lr time:',lr_time)

    gcn_time=time.time()
    # print("****************this is drug fea1******************")
    # print(X_drug_feat_data_train1.shape)
    # print("****************this is drug adj1******************")
    # print(X_drug_adj_data_train1)
    # print("****************this is drug fea2******************")
    # print(X_drug_feat_data_train2)
    # print("****************this is drug adj2******************")
    # print(X_drug_adj_data_train2)
    GCNModel.fit(x=[X_drug_feat_data_train1, X_drug_adj_data_train1, X_drug_feat_data_train2, X_drug_adj_data_train2],y=Y_train, batch_size=64, epochs=nb_epoch, validation_split=0, callbacks=callbacks)
    gcn_time=time.time()-gcn_time
    print('gcn time:',gcn_time)

    rfc_time=time.time()
    rfc.fit(x, Y_train)
    rfc_time=time.time()-rfc_time
    print('rfc time:',rfc_time)
    print>>doc,'lr time:',lr_time
    print>>doc,'rfc time:',rfc_time
    print>>doc,'gcn time:',gcn_time
    with open('./trained_models/lr.pickle', 'wb') as fw:
        pickle.dump(lrModel, fw)
    with open('./trained_models/rfc.pickle', 'wb') as frfc:
        pickle.dump(rfc, frfc)
    return lrModel, rfc, GCNModel


def ModelEvaluate(result_save_path, Max_atoms, doc, lr, rfc, model, X_drug_feat_data_test1, X_drug_adj_data_test1, X_drug_feat_data_test2,
                  X_drug_adj_data_test2, Y_test):
    random.seed(0)
    x_test = np.c_[X_drug_feat_data_test1.reshape(-1, 75*Max_atoms), X_drug_adj_data_test1.reshape(-1, Max_atoms*Max_atoms),X_drug_feat_data_test2.reshape(-1, 75*Max_atoms), X_drug_adj_data_test2.reshape(-1, Max_atoms*Max_atoms)]
    y_pred_val = model.predict([X_drug_feat_data_test1, X_drug_adj_data_test1,
                                X_drug_feat_data_test2, X_drug_adj_data_test2])
    y_pred_val_lr = lr.predict_proba(x_test)
    y_pred_val_lr = y_pred_val_lr[:,1]


    y_pred_val_rfc = rfc.predict_proba(x_test)
    y_pred_val_rfc = y_pred_val_rfc[:,1]

    precision1, recall1, thresholds1 = precision_recall_curve(Y_test, y_pred_val)
    average_precision = average_precision_score(Y_test, y_pred_val)

    precision2, recall2, thresholds2 = precision_recall_curve(Y_test, y_pred_val_lr)
    average_precision2 = average_precision_score(Y_test, y_pred_val_lr)

    precision3, recall3, thresholds3 = precision_recall_curve(Y_test, y_pred_val_rfc)
    average_precision3 = average_precision_score(Y_test, y_pred_val_rfc)

    np.savez('%slr_gcn_rfc_svm_pr_result.npz' % result_save_path, gcn_fpr=precision1, gcn_tpr=recall1,
                                    lr_fpr=precision2, lr_tpr=recall2,
                                    rfc_fpr=precision3, rfc_tpr=recall3)

    np.savez('%sy_pred.npz' % result_save_path, y=Y_test, GCN=y_pred_val,
                                LR=y_pred_val_lr,
                                RFC=y_pred_val_rfc)

    p_r_curve = plt.figure()
    print('the ap of GCN: %f' % average_precision)
    print('the ap of LR: %f' % average_precision2)
    print('the ap of RFC: %f' % average_precision3)

    plt.plot(recall1, precision1, label='PRC of GCN (AUPRC = %f)' % average_precision)
    plt.plot(recall2, precision2, label='PRC of LR (AUPRC = %f)' % average_precision2)
    plt.plot(recall3, precision3, label='PRC of RF (AUPRC = %f)' % average_precision3)

    print >> doc,'the ap of GCN: %f' % average_precision
    print >> doc,'the ap of LR: %f' % average_precision2
    print >> doc,'the ap of RFC: %f' % average_precision3

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('total Precision-Recall curve')
    plt.legend(loc='upper right')
    plt.show(p_r_curve)
    plt.savefig('%sPRC_GCN_LR_RFC_SVM.png' % result_save_path)
    # return average_precision

    roc = plt.figure()
    auc_score = roc_auc_score(Y_test, y_pred_val)
    print('the auc score of GCN is : %s' % auc_score)
    print >> doc,'the auc score of GCN is : %s' % auc_score
    fpr1, tpr1, thresholds1 = roc_curve(Y_test, y_pred_val)  # roc1
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, lw=1, alpha=0.3, label='ROC of GCN (AUC = %f)' % roc_auc1)

    # evaluate lr model
    auc_score_lr = roc_auc_score(Y_test, y_pred_val_lr)
    print('the auc of lr is : %s' % auc_score_lr)
    print >> doc,'the auc of LR is : %s' % auc_score_lr
    fpr2, tpr2, thresholds2 = roc_curve(Y_test, y_pred_val_lr)
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, lw=1, alpha=0.3, label='ROC of LR (AUC = %f)' % roc_auc2)

    # evaluate rfc model
    auc_score_rfc = roc_auc_score(Y_test, y_pred_val_rfc)
    print('the auc of rf is : %s' % auc_score_rfc)
    print >> doc,'the auc of RFC is : %s' % auc_score_rfc
    fpr3, tpr3, thresholds3 = roc_curve(Y_test, y_pred_val_rfc)
    roc_auc3 = auc(fpr3, tpr3)
    plt.plot(fpr3, tpr3, lw=1, alpha=0.3, label='ROC of RF (AUC = %f)' % roc_auc3)
    
    np.savez('%slr_gcn_rfc_svm_roc_result.npz' % result_save_path, gcn_fpr=fpr1, gcn_tpr=tpr1,
                                                        lr_fpr=fpr2, lr_tpr=tpr2,
                                                        rfc_fpr=fpr3, rfc_tpr=tpr3)

    # plot together
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show(roc)
    plt.savefig("%sROC_GCN_LR_RFC_SVM.png" % result_save_path)


def train(location, Max_atoms, gpu_id, database, sample, unit_list, use_bn, use_relu, use_GMP):
    random.seed(0)
    doc = set_data(database, sample, unit_list, use_bn, use_relu, use_GMP, location)
    result_save_path='/home/zengwanwen/caoxusheng/DeepDDI_desktop/results/'+ \
        database + '/' + sample+'/'+ location+'/'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print("Begin to load")
    print>>doc,'Begin to load'
    data_path = '/home/zengwanwen/caoxusheng/DeepDDI_desktop/data/preprocessed_data/'+ database + '/' + sample +'/2c'

    train_data = np.load('%s/train_data.npz' % (data_path))
    test_data = np.load('%s/test_data.npz' % (data_path))

    print("done loading!")
    print>>doc,'done loading'
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    print("begin split")
    print>>doc,'begin split'
    X_train, X_val, y_train, y_val = DataSplit(X_train, y_train)
    print("done split!")
    print>>doc,'done split'

    X_test = test_data['X_test']
    y_test = test_data['y_test']


    lr = LogisticRegression(random_state=0, dual=False, solver='lbfgs')
    rfc = RandomForestClassifier(n_estimators=100, random_state=0,max_depth=2)
    
    np.set_printoptions(threshold=np.inf)
    tmp1 = 75*Max_atoms
    tmp2 = tmp1+Max_atoms*Max_atoms
    tmp3 = tmp2+75*Max_atoms
    tmp4 = tmp3+Max_atoms*Max_atoms
    
    
    X_drug_feat_data_train1 = X_train[:, :tmp1].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_train1 = np.zeros([X_drug_feat_data_train1.shape[0], Max_atoms, Max_atoms])
    X_drug_feat_data_train2 = X_train[:, tmp2:tmp3].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_train2 = np.zeros([X_drug_feat_data_train2.shape[0], Max_atoms, Max_atoms])

    X_drug_feat_data_test1 = X_test[:, :tmp1].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_test1 = np.zeros([X_drug_feat_data_test1.shape[0], Max_atoms, Max_atoms])
    X_drug_feat_data_test2 = X_test[:, tmp2:tmp3].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_test2 = np.zeros([X_drug_feat_data_test2.shape[0], Max_atoms, Max_atoms])

    X_drug_feat_data_val1 = X_val[:, :tmp1].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_val1 = np.zeros([X_drug_feat_data_val1.shape[0], Max_atoms, Max_atoms])
    X_drug_feat_data_val2 = X_val[:, tmp2:tmp3].reshape(-1, Max_atoms, 75)
    X_drug_adj_data_val2 = np.zeros([X_drug_feat_data_val2.shape[0], Max_atoms, Max_atoms])

    validation_data = [[X_drug_feat_data_val1, X_drug_adj_data_val1,X_drug_feat_data_val2, X_drug_adj_data_val2], y_val]
    model = KerasMultiSourceGCNModel(False, False, False, False).createMaster(X_drug_feat_data_test1[0].shape[-1], unit_list, use_relu, use_bn, use_GMP)

    lr, rfc, model = ModelTraining(Max_atoms, doc, sample, lr, rfc, model, X_drug_feat_data_train1, X_drug_adj_data_train1, \
        X_drug_feat_data_train2, X_drug_adj_data_train2, y_train, validation_data, nb_epoch=100)
    ModelEvaluate(result_save_path, Max_atoms, doc, lr, rfc, model, X_drug_feat_data_test1, X_drug_adj_data_test1,
                X_drug_feat_data_test2, X_drug_adj_data_test2, y_test)
    doc.close()
