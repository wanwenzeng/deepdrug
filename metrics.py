from copy import deepcopy
from collections import defaultdict,Counter,OrderedDict
from functools import reduce
import pickle as pkl
from itertools import product
import warnings
import numpy as np
import yaml
import pandas as pd
import random,time,datetime,json
warnings.filterwarnings('ignore')
from typing import Optional, List, NamedTuple
import matplotlib.pyplot as plt
import argparse
from time import sleep 
import torch
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch import nn 
import os,sys
from sklearn.metrics import (accuracy_score,auc,average_precision_score,f1_score,
                                precision_recall_curve,precision_score,recall_score,
                                roc_auc_score,roc_curve,classification_report,r2_score,
                                explained_variance_score)
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

pathjoin = lambda *args,**argvs : os.path.join(*args,**argvs)

t2np = lambda t: t.detach().cpu().numpy()


def to_categorical_func(y, num_classes=None, dtype='float32'):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def evaluate_binary(y_true,y_pred,):
    
    y_true = y_true.astype(int)
    y_pred = y_pred.reshape(-1)
    y_pred_cls = (y_pred>=0.5).astype(int)
    metric_dict = {}
    metric_dict['y_true'] = y_true
    metric_dict['y_pred'] = y_pred
    metric_dict['y_true_cls'] = y_true
    metric_dict['y_pred_cls'] = y_pred_cls
    metric_dict['F1']  = f1_score(y_true,y_pred_cls,average='binary')
    metric_dict['Acc']  = accuracy_score(y_true,y_pred_cls)
    metric_dict['prc_prec'] ,metric_dict['prc_recall'] ,metric_dict['prc_thres'] = precision_recall_curve(y_true,y_pred)
    metric_dict['roc_tpr'] ,metric_dict['roc_fpr'] ,metric_dict['roc_thres'] = roc_curve(y_true,y_pred)
    metric_dict['auROC'] = auc(metric_dict['roc_tpr'],metric_dict['roc_fpr'])
    metric_dict['auPRC'] = auc(metric_dict['prc_recall'],metric_dict['prc_prec'])

    return metric_dict

def evaluate_multiclass(y_true,y_pred,to_categorical=False,num_classes=None):

    if to_categorical:
        y_true_cls = y_true.copy()
        # print('y_true_cls',y_true_cls[:10])
        y_true = to_categorical_func(y_true,num_classes)
    else:
        y_true_cls = y_true.argmax(axis=1)

    y_pred_cls = y_pred.argmax(axis=1)


    metric_dict = {}
    metric_dict['y_true'] = y_true
    metric_dict['y_pred'] = y_pred
    metric_dict['y_true_cls'] = y_true_cls
    metric_dict['y_pred_cls'] = y_pred_cls
    
    metric_dict['F1-macro']  = f1_score(y_true_cls,y_pred_cls,average='macro')
    metric_dict['F1-micro']  = f1_score(y_true_cls,y_pred_cls,average='micro')
    metric_dict['F1'] = metric_dict['F1-macro'] 

    for idx in np.arange(num_classes):
        (metric_dict['prc_prec@%d'%(idx)],
            metric_dict['prc_recall@%d'%(idx)],
            metric_dict['prc_thres@%d'%(idx)]) = precision_recall_curve(y_true[:,idx],y_pred[:,idx])
        (metric_dict['roc_tpr@%d'%(idx)],
            metric_dict['roc_fpr@%d'%(idx)],
            metric_dict['roc_thres@%d'%(idx)]) = roc_curve(y_true[:,idx],y_pred[:,idx])
        metric_dict['auROC@%d'%(idx)] = auc(metric_dict['roc_tpr@%d'%(idx)],
                                            metric_dict['roc_fpr@%d'%(idx)])
        metric_dict['auPRC@%d'%(idx)] = auc(metric_dict['prc_recall@%d'%(idx)],
                                            metric_dict['prc_prec@%d'%(idx)],
                                            )

    metric_dict['auPRC'] = pd.Series([ metric_dict['auPRC@%d'%(idx)] for idx in np.arange(num_classes) ]).fillna(0).mean()
    metric_dict['auROC'] = pd.Series([ metric_dict['auROC@%d'%(idx)] for idx in np.arange(num_classes) ]).fillna(0).mean()
    return metric_dict

def evaluate_multilabel(y_true,y_pred,thers=0.5,verbose=False):
    y_true = y_true.astype(int)
    y_pred_cls = (y_pred>=thers).astype(int)
    # print(y_true[0])
    # print(y_pred_cls[0])
    num_classes = y_true.shape[-1]
    metric_dict = {}
    metric_dict['y_true'] = y_true
    metric_dict['y_pred'] = y_pred
    metric_dict['y_true_cls'] = y_true
    metric_dict['y_pred_cls'] = y_pred_cls

    metric_dict['F1-macro-old']  = f1_score(y_true,y_pred_cls,average='macro')
    metric_dict['F1-micro-old'] = f1_score(y_true.reshape(-1),y_pred_cls.reshape(-1),average='micro')
    # metric_dict['F1'] = metric_dict['F1-macro'] 

    if verbose:
        loops = tqdm(np.arange(num_classes))
    else:
        loops = np.arange(num_classes)
    for idx in loops :
        metric_dict['F1@%d'%(idx)] = f1_score(y_true[:,idx],y_pred_cls[:,idx])

        (metric_dict['prc_prec@%d'%(idx)],
            metric_dict['prc_recall@%d'%(idx)],
            metric_dict['prc_thres@%d'%(idx)]) = precision_recall_curve(y_true[:,idx],y_pred[:,idx])
        (metric_dict['roc_tpr@%d'%(idx)],
            metric_dict['roc_fpr@%d'%(idx)],
            metric_dict['roc_thres@%d'%(idx)]) = roc_curve(y_true[:,idx],y_pred[:,idx])
        metric_dict['auROC@%d'%(idx)] = auc(metric_dict['roc_tpr@%d'%(idx)],
                                            metric_dict['roc_fpr@%d'%(idx)])
        metric_dict['auPRC@%d'%(idx)] = auc(metric_dict['prc_recall@%d'%(idx)],
                                            metric_dict['prc_prec@%d'%(idx)])

    metric_dict['auPRC'] = pd.Series([ metric_dict['auPRC@%d'%(idx)] for idx in np.arange(num_classes) ]).fillna(0).mean()
    metric_dict['auROC'] = pd.Series([ metric_dict['auROC@%d'%(idx)] for idx in np.arange(num_classes) ]).fillna(0).mean()
    metric_dict['F1-macro'] = pd.Series([ metric_dict['F1@%d'%(idx)] for idx in np.arange(num_classes) ]).fillna(0).mean()
    metric_dict['F1'] = metric_dict['F1-macro']
    return metric_dict


def f1_score_thread_func(inputs):
    thres, y_true,y_pred = inputs
    y_pred = (y_pred >=thres).astype(int)
    return f1_score(y_true,y_pred)
from multiprocessing import Pool



def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))


def evaluate_regression(y_true,y_pred):
    metric_dict = {}
    metric_dict['y_true'] = y_true.reshape(-1)
    metric_dict['y_pred'] = y_pred.reshape(-1)
    metric_dict['r2'] = r2_score(metric_dict['y_true'], metric_dict['y_pred'],)
    metric_dict['mse'] =  ((metric_dict['y_true']- metric_dict['y_pred'])**2).mean()
    pr,pr_p_val = pearsonr(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['pearsonr'] = pr
    metric_dict['pearsonr_p_val'] = pr_p_val
    metric_dict['concordance_index']= concordance_index(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['explained_variance'] = explained_variance_score(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['cindex'] = -1 #get_cindex(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['rm2'] = get_rm2(metric_dict['y_true'].reshape(-1), metric_dict['y_pred'].reshape(-1))
    return metric_dict
