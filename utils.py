import numpy as np 

from collections import  OrderedDict
import torch 
import torch as t
import os,json
class Struct():
    def __init__(self,**argvs):
        for k,v in argvs.items():
            setattr(self,k,v)


from torch.nn.init import uniform_,zeros_,xavier_normal_
@torch.no_grad()
def init_linear(m):
    # print(m)
    if type(m) == nn.Linear:
        xavier_normal_(m.weight,gain=1)
        if m.bias is not None :
            zeros_(m.bias)
        else:
            pass 


def y_log10_transfrom_func(y):
    '''
    equal to convert_y_unit(y,'nM','p') in DeepPurpose
    '''
    print('log10 transfromation for targets')
    zero_idxs = np.where(y <= 0.)[0]
    y[zero_idxs] = 1e-10
    y = -np.log10(y*1e-9)
    return y.astype(float)

def y_kiba_transform_func(y):
    y = -y 

    return np.abs(np.min(y))+y
def print_args(**kwargs):
    print('print parameters:')
    args_dict = OrderedDict()
    for k,v in sorted(kwargs.items(), key=lambda x: x[0]):
        if isinstance(v,(str,float,int,list,type(None))) :
            args_dict[k]=v
        else:
            print(k,v)
    print(json.dumps(args_dict,indent=2))
    return kwargs
pathjoin =  os.path.join

t2np = lambda t: t.detach().cpu().numpy()



def keep_scalar_func(in_dict,prefix=''):
    if prefix != '': prefix += '_'
    out_dict = {}
    for k,v in in_dict.items():

        if isinstance(v,(t.Tensor)):
            v = t2np(v) 
        if isinstance(v,(float,int)):
            out_dict[prefix+k]=v

    return out_dict

