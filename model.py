import numpy as np 
import pandas as pd 
# from copy import deepcopy
from collections import OrderedDict
# from functools import reduce
# import pickle as pkl
# from itertools import product
import warnings
warnings.filterwarnings('ignore')
import json
from typing import Optional, List, NamedTuple
import torch
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch import nn 
# from torch.nn import ModuleList, BatchNorm1d
from torch.autograd import Variable
from typing import Union, Tuple,Optional
from torch_geometric.typing import OptPairTensor, Adj, Size,OptTensor
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_softmax
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool
# import  torch_geometric.nn as gnn
from dataset import (smile_dict,trans_seqs,seq_dict)
from pytorch_lightning import LightningModule
from utils import * 
from metrics import  (evaluate_binary,evaluate_multiclass,evaluate_multilabel,evaluate_regression)


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


def structDict(**argvs):
    return argvs
from torch.nn.init import uniform_,zeros_,xavier_normal_
@torch.no_grad()
def init_linear(m):
    if type(m) == nn.Linear:
        xavier_normal_(m.weight,gain=1)
        if m.bias is not None :
            zeros_(m.bias)
        else:
            pass 


import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MyCrossEntropyLoss(t.nn.Module):
    def __init__(self,weight = None, size_average=None, ignore_index = -100, reduce=None, reduction = 'mean'):
        super(MyCrossEntropyLoss, self).__init__()
        self.loss = t.nn.NLLLoss(weight,size_average,ignore_index,reduce,reduction)
    
    def forward(self,output,target):
        '''
        output : [N,C], predicted probability of sample belonging to each classes,i.e., values after activation functions (softmax)
                target: [N,],
        
        '''
        output = t.log(output+ 1e-10)
        return self.loss(output,target)



# modify from https://github.com/rusty1s/pytorch_geometric
class DeepGCNLayerV2(torch.nn.Module):
    def __init__(self, conv=None, norm=None, act=None, block='res+', 
                 dropout=0., ckpt_grad=False,edge_norm = None,):
        super(DeepGCNLayerV2, self).__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.edge_norm  = edge_norm
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad


    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs):
        """"""
        args = list(args)
        x = args.pop(0)
        org_edge_attr = args[1] #org_edge_attr:[edge_index,edge_attr,....]
        if org_edge_attr is None: org_edge_attr = 0
        if self.block == 'res+':
            if self.norm is not None:
                h = self.norm(x)
            if self.act is not None:
                h = self.act(h)
            if self.edge_norm is not None:
                args[1] = self.edge_norm(args[1])
                # need to uncomment
                # if self.act is not None:
                #     args[1]  = self.act(args[1] )
                # args[1]  = F.dropout(args[1] , p=self.dropout, training=self.training)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h,edge_attr = checkpoint(self.conv, h, *args, **kwargs)
            else:
                h,edge_attr = self.conv(h, *args, **kwargs)

            return x + h,org_edge_attr +edge_attr 

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h,edge_attr = checkpoint(self.conv, x, *args, **kwargs)
            else:
                h,edge_attr = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            if self.edge_norm is not None:
                edge_attr = self.edge_norm(edge_attr)

            if self.block == 'res':
                return x + h, org_edge_attr+edge_attr
            elif self.block == 'dense':
                return torch.cat([x, h], dim=-1),org_edge_attr+edge_attr
            elif self.block == 'plain':
                return h,edge_attr

    def __repr__(self):
        return '{}(block={})'.format(self.__class__.__name__, self.block)

# modify from https://github.com/rusty1s/pytorch_geometric
class SAGEConvV2(MessagePassing):
    # concat edge_index and node features
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 in_edge_channels: Union[int, Tuple[int, int],None] = None ,
                 aggr: str = 'mean', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False,
                  **kwargs):  # yapf: disable
        # kwargs.setdefault('aggr', 'mean')
        kwargs.setdefault('aggr', None)
        super(SAGEConvV2, self).__init__(**kwargs)

        self.aggr = aggr
        self.eps = 1e-7
        assert aggr in ['softmax', 'softmax_sg', 'power','mean','add','sum']
        if self.aggr in ['softmax', 'softmax_sg', 'power',]:
            self.initial_t = t
            if learn_t and aggr == 'softmax':
                self.t = Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.t = t
            self.initial_p = p
            if learn_p:
                self.p = Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels 
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        
        if in_edge_channels is not None:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0]*2 + in_edge_channels, in_channels[0]*2 , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0]*2 , out_channels, bias=bias),
                                        )   
        else:
            self.lin_l = nn.Sequential(
                                        nn.Linear(in_channels[0]*2 , in_channels[0]*2 , bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(in_channels[0]*2 , out_channels, bias=bias),
                        )
        self.reset_parameters()

    def reset_parameters(self,):
        # self.lin_l.reset_parameters()
        self.lin_l.apply(init_linear)
        
        if self.root_weight:
            self.lin_r.reset_parameters()

        # if self.in_edge_channels is not None:
        #     self.edge_lin.reset_parameters()

        if self.aggr in ['softmax', 'softmax_sg', 'power',]:
            if self.t and isinstance(self.t, Tensor):
                self.t.data.fill_(self.initial_t)
            if self.p and isinstance(self.p, Tensor):
                self.p.data.fill_(self.initial_p)


    def message(self, x_i: Tensor,x_j: Tensor,edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            x = t.cat([x_i,x_j],dim=-1)
        else:
            x = t.cat([x_i,x_j,edge_attr],dim=-1)
        x  = self.lin_l(x)
        return x
    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggr == 'softmax':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum'),inputs

        elif self.aggr == 'softmax_sg':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum'),inputs

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p),inputs

        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                               reduce=self.aggr),inputs
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, 
                edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out,edge_attr  = self.propagate(edge_index, x=x, edge_attr=edge_attr,size=size)


        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out,edge_attr

class DeeperGCN(nn.Module):
    def __init__(self, in_channel,mid_channel, num_layers,num_blocks = 1, dropout_ratio=0.1,embedding_layer=None,graph_conv=SAGEConvV2,
                in_edge_channel=None,node_encoding=True,aggr='softmax'):
        super(DeeperGCN, self).__init__()
        self.update_attr = False

            
        if graph_conv in [SAGEConvV2,]:
            if aggr == 'softmax':
                graph_para_dict = {'in_edge_channels':mid_channel,
                                    'aggr': 'softmax', "t":  1.0, 'learn_t':  True,
                                    }
            elif aggr in ['mean','add','sum']:
                graph_para_dict = {'in_edge_channels':mid_channel,'aggr':aggr}

            use_edge_encoder = True 
            self.update_attr = True 
            

        else:
            raise 
        # graph_para_dict = {}
        self.embedding_layer = embedding_layer

        in_channel = in_channel if self.embedding_layer is None else self.embedding_layer.embedding_dim
        self.dropout_ratio = dropout_ratio

        
        if node_encoding:
            self.node_encoder = nn.Sequential(
                                    nn.Linear(in_channel,mid_channel),
                                    nn.LayerNorm(mid_channel),
                                    )
        else:
            self.node_encoder = None 


        self.gcn_blocks = nn.ModuleList()

        for block in range(num_blocks):
            layers = nn.ModuleList()
            for i in range(1, num_layers + 1):
                conv = graph_conv(mid_channel, mid_channel,**graph_para_dict )
                norm = nn.LayerNorm(mid_channel, elementwise_affine=True)
                edge_norm = nn.LayerNorm(mid_channel, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayerV2(conv, norm, act, block='res+', dropout=dropout_ratio,
                                     # ckpt_grad=  False,
                                     ckpt_grad=  i % 3,edge_norm =edge_norm,
                                     )
                layers.append(layer)
            self.gcn_blocks.append(layers)

        self.edge_encoder = None 
        self.use_attr = True if in_edge_channel is not None else False 
        if self.use_attr and use_edge_encoder:
            self.edge_encoder = nn.Sequential(
                                    nn.Linear(in_edge_channel,mid_channel),
                                    nn.LayerNorm(mid_channel),
                                    )


    def forward(self, x, edge_index,edge_attr=None,batch=None):
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        out = [] 
        for gcn_block in self.gcn_blocks:
            if self.use_attr:  
                x = gcn_block[0].conv(x, edge_index,edge_attr)
                if self.update_attr: x,edge_attr = x
                for layer in gcn_block[1:]:
                    x = layer(x, edge_index,edge_attr)  
                    if self.update_attr: x,edge_attr = x     
            else: 
                x = gcn_block[0].conv(x, edge_index)
                if self.update_attr: x,__edge_attr = x  #edge_attr is None
                for layer in gcn_block[1:]:
                    x = layer(x, edge_index) 
                    if self.update_attr: x,__edge_attr = x
            x = gcn_block[0].act(gcn_block[0].norm(x))
            # if self.update_attr: x,edge_attr = x
            out.append(x)

        return t.cat(out,dim=1)

class CNN(nn.Sequential):
    def __init__(self,in_channel,mid_channel,seq_len,dropout_ratio=0.1):
        super(CNN, self).__init__()
        self.seq_len= seq_len 
        in_channel = in_channel

        encoding = 'drug'
        config = structDict( 
                         cls_hidden_dims = [1024,1024,512], 
                         cnn_drug_filters = [32,64,96],
                         cnn_target_filters = [32,64,96],
                         cnn_drug_kernels = [4,6,8],
                         cnn_target_kernels = [4,8,12]
                        )
        if encoding == 'drug':
            in_ch = [in_channel] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output(( in_channel,seq_len,))
            #n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, mid_channel)

        if encoding == 'protein':
            in_ch = [in_channel] + config['cnn_target_filters']
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i], 
                                                    out_channels = in_ch[i+1], 
                                                    kernel_size = kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output(( in_channel,seq_len,))

            self.fc1 = nn.Linear(n_size_p, mid_channel)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):

        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v



class DeepDrug(nn.Module):
    def __init__(self,in_channel=93,mid_channel=128,num_out_dim=1,out_activation_func = 'softmax',
        siamese_feature_module = True,dropout_ratio=0.1,entry2_in_channel=None,entry2_mid_channel=128,num_graph_layer=22,
                entry1_seq_len=200,entry2_seq_len=None,
                entry2_in_edge_channel=None,entry2_num_graph_layer=None ,in_edge_channel=11,
                
                **kwargs
                ):
        super(DeepDrug,self).__init__()
        if kwargs.get('verbose',True):
            print_args(**kwargs)
        assert out_activation_func in ['softmax','sigmoid',None]

        


        self.siamese_feature_module = siamese_feature_module
        self.out_activation_func = out_activation_func 
        self.dropout_ratio = dropout_ratio

        self.num_graph_layer = num_graph_layer 


        self.gconv1 = DeeperGCN(in_channel,mid_channel, self.num_graph_layer,1,dropout_ratio=0.1,embedding_layer=None,graph_conv=SAGEConvV2,in_edge_channel=in_edge_channel,
                                aggr='softmax')
        dim_gconv1_out = mid_channel  *1

        self.gconv1_seq = CNN( len(smile_dict),mid_channel,seq_len= entry1_seq_len)
        dim_gconv1_seq_out = mid_channel
        dim_gconv1_out +=  dim_gconv1_seq_out  
            
        if self.siamese_feature_module:
            print('siamese_feature_module setting...')
            self.gconv2 = self.gconv1 
            dim_gconv2_out = dim_gconv1_out
            self.gconv2_seq = self.gconv1_seq

        else:
            #entry2 graph module 
            
            if (entry2_in_channel is None) or(entry2_mid_channel is None):
                self.gconv2 = DeeperGCN(entry2_in_channel,entry2_mid_channel, entry2_num_graph_layer,1,dropout_ratio=0.1,embedding_layer=None,graph_conv=SAGEConvV2,
                                in_edge_channel=entry2_in_edge_channel,
                                aggr='softmax')
                dim_gconv2_out = entry2_mid_channel  *1     
            else :
                entry2_mid_channel = mid_channel if entry2_mid_channel is None else entry2_mid_channel
                entry2_in_channel = in_channel if entry2_in_channel is None else entry2_in_channel
                self.gconv2 = DeeperGCN(entry2_in_channel,entry2_mid_channel, entry2_num_graph_layer,1,dropout_ratio=0.1,embedding_layer=None,graph_conv=SAGEConvV2,
                                in_edge_channel=entry2_in_edge_channel,
                                aggr='softmax')
                dim_gconv2_out = entry2_mid_channel  *1

                self.gconv2_seq = CNN(len(seq_dict) ,entry2_mid_channel,seq_len= entry2_seq_len)
                dim_gconv2_seq_out = entry2_mid_channel
                dim_gconv2_out +=  dim_gconv2_seq_out  

        channel_list = [dim_gconv1_out +dim_gconv2_out ,]+[128,32] 
        latent_dim = channel_list[-1]
        nn_list = []
        for idx,num in enumerate(channel_list[:-1]):
            nn_list.append(nn.Linear(channel_list[idx],channel_list[idx+1]))
            nn_list.append(nn.BatchNorm1d(channel_list[idx+1]))
            if self.dropout_ratio >0:
                nn_list.append(nn.Dropout(self.dropout_ratio))
            nn_list.append(nn.ReLU())
        self.global_fc_nn =nn.Sequential(*nn_list)    
        self.fc2 = nn.Linear(latent_dim,num_out_dim)

        self.reset_parameters()

    def reset_parameters(self,):
        self.apply(init_linear) 

    def forward(self,entry1_data,entry2_data,get_latent_varaible=False):

        entry2_data,entry2_seq_data = entry2_data
        entry1_data,entry1_seq_data = entry1_data

        entry1_x,entry1_edge_index,entry1_edge_attr,entry1_batch = entry1_data.x,entry1_data.edge_index,entry1_data.edge_attr,entry1_data.batch
        entry1_out = self.gconv1(entry1_x,entry1_edge_index,entry1_edge_attr,entry1_batch )
        entry1_mean = global_mean_pool(entry1_out,entry1_batch)
        entry1_seq_mean = self.gconv1_seq(entry1_seq_data)
        entry1_mean = t.cat([entry1_mean,entry1_seq_mean],dim=-1)

        entry2_x,entry2_edge_index,entry2_edge_attr,entry2_batch = entry2_data.x,entry2_data.edge_index,entry2_data.edge_attr,entry2_data.batch
        entry2_out = self.gconv2(entry2_x,entry2_edge_index,entry2_edge_attr,entry2_batch)
        entry2_mean = global_mean_pool(entry2_out,entry2_batch)
        entry2_seq_mean = self.gconv2_seq(entry2_seq_data)
        entry2_mean = t.cat([entry2_mean,entry2_seq_mean],dim=-1)

        cat_features = t.cat([entry1_mean,entry2_mean],dim=-1)
        x = self.global_fc_nn(cat_features)  
        if get_latent_varaible:
            return x
        else:
            x = self.fc2(x)
            if self.out_activation_func == 'softmax':
                return F.softmax(x, dim=-1) # F.log_softmax(x, dim=-1)
            elif self.out_activation_func == 'sigmoid':
                return t.sigmoid(x)
            elif self.out_activation_func is None : 
                return x 


class DeepDrug_Container(LightningModule):
    def __init__(self,num_out_dim=1,
                 task_type = 'multi_classification',lr = 0.001,category = None , verbose=True ,my_logging=False, scheduler_ReduceLROnPlateau_tracking='mse'
                 ):
        super().__init__()

        self.save_hyperparameters()
        # self.save_hyperparameters()
        assert task_type in ['regression','binary_classification','binary',
                            'multi_classification','multilabel_classification','multiclass','multilabel',
                             ]

        self.verbose = verbose
        self.my_logging = my_logging
        self.scheduler_ReduceLROnPlateau_tracking = scheduler_ReduceLROnPlateau_tracking
        self.lr = lr 
        self.task_type = task_type
        self.num_classes = num_out_dim

        self.category = category
        if self.category == 'DDI':
            self.entry2_type= 'drug' 
            self.entry2_num_graph_layer= 22
            self.entry2_seq_len= 200
            self.entry2_in_channel=  91 + 2 
            self.entry2_in_edge_channel= 11 
            self.siamese_feature_module=True  
            
        elif self.category == 'DTA':
            self.siamese_feature_module=False 
            self.entry2_type = 'protein'
            self.entry2_num_graph_layer= 6
            self.entry2_seq_len=1000
            self.entry2_in_channel= 78 + 2 
            self.entry2_in_edge_channel= 2
        else:raise 

        if self.task_type in ['multi_classification','multiclass']:
            out_activation_func = 'softmax'
            self.loss_func =MyCrossEntropyLoss()
        elif self.task_type in ['binary_classification','binary','multilabel_classification','multilabel']:
            out_activation_func = 'sigmoid'
            self.loss_func = F.binary_cross_entropy
        elif self.task_type in ['regression',]:
            out_activation_func = None
            self.loss_func = F.mse_loss 
        else:
            raise 
        
        self.model = DeepDrug(num_out_dim=num_out_dim,out_activation_func = out_activation_func,siamese_feature_module = self.siamese_feature_module,
                            entry2_in_channel=self.entry2_in_channel, entry2_seq_len = self.entry2_seq_len, entry2_in_edge_channel= self.entry2_in_edge_channel,
                            entry2_num_graph_layer = self.entry2_num_graph_layer,
                            )


        if self.verbose: print(self.model,)
        self.epoch_metrics = Struct(train=[],valid=[],test=[])  
        self.metric_dict = {}
    
    def forward(self,batch) :
        (entry1_data,entry2_data),y  = batch
        return self.model(entry1_data,entry2_data)
    def training_step(self, batch, batch_idx):
        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)

        # print('\nloss',loss,y_out.shape,y.shape,y_out.dtype,y.dtype,y_out[:2],y[:2])
        self.log('train_loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True)
        lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        self.log('lr', np.round(lr,6), prog_bar=True, on_step=True,
                 on_epoch=False)


        return_dict = {'loss':loss,'y_out':t2np(y_out),'y':t2np(y)}

        return return_dict 

    def validation_step(self, batch, batch_idx):

        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)

        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True)

        # return loss 
        return_dict = {'y_out':t2np(y_out),'y':t2np(y)}
        return return_dict  

    def test_step(self, batch, batch_idx):
        (entry1_data,entry2_data),y  = batch
        y_out = self(batch)
        if self.task_type in ['multi_classification','multiclass',]:
            loss = self.loss_func(y_out, y.reshape(-1))
        elif self.task_type in ['binary','binary_classification']:
            loss = self.loss_func(y_out, y.float())
        elif self.task_type in ['multilabel_classification','multilabel',]:
            loss = self.loss_func(y_out, y.float())
        else:
            loss = self.loss_func(y_out, y)
        self.log('test_loss', loss, prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
        return_dict = {'y_out':t2np(y_out),'y':t2np(y)} 
        return return_dict 

    def training_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])


            
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'trn',)
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_trn'))
        try: self.epoch_metrics.train.pop(-1)
        except: pass
        self.epoch_metrics.train.append(metric_dict)



    def validation_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'val')
        if self.task_type in ['binary','multiclass','multilabel']:
            self.log('val_epoch_F1', metric_dict['F1'], prog_bar=False, on_step=False,on_epoch=True)
            self.log('val_epoch_auPRC', metric_dict['auPRC'], prog_bar=False, on_step=False,on_epoch=True)
        elif self.task_type in ['regression',]:
            self.log('val_epoch_MSE', metric_dict['mse'], prog_bar=False, on_step=False,on_epoch=True)

        try: self.epoch_metrics.valid.pop(-1)
        except: pass
        self.epoch_metrics.valid.append(metric_dict)
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_val'))

        if (len(self.epoch_metrics.train)>0) :
            self.print_metrics_on_epoch_end(self.epoch_metrics.train[-1])
        self.print_metrics_on_epoch_end(self.epoch_metrics.valid[-1])
          
        self.my_schedulers.step(metric_dict[self.scheduler_ReduceLROnPlateau_tracking])  #'mse','F1'



    def test_epoch_end(self,outputs):
        y_out  = np.concatenate([x['y_out'] for x in outputs])
        y  = np.concatenate([x['y'] for x in outputs])
        metric_dict = self.cal_metrics_on_epoch_end(y,y_out,'tst')
        
        if self.my_logging:
            self.logger.log_metrics(keep_scalar_func(metric_dict,prefix='epoch_tst'))

        try: self.epoch_metrics.test.pop(-1)
        except: pass
        self.epoch_metrics.test.append(metric_dict)

        self.print_metrics_on_epoch_end(self.epoch_metrics.test[-1])



    def print_metrics_on_epoch_end(self,metric_dict):
        try:
            lr = iter(self.my_optimizers.param_groups).__next__()['lr']
        except:
            lr = 0

        if 'F1' in metric_dict.keys():
            print('\n%s:Ep%04d|| F1: %.03f,auROC %.03f,auPRC: %.03f'%(metric_dict['prefix'],metric_dict['epoch'],metric_dict['F1'],
                                        metric_dict['auROC'],
                                        metric_dict['auPRC'],))
        elif 'mse' in metric_dict.keys():
            print('\n%s:Ep%04d|| MSE: %.03f,lr: %.07f,r2: %.03f,pear-r: %.03f,con_index: %.03f,expVar: %.03f,cidx: %.03f,rm2: %.03f'%(
                                        metric_dict['prefix'],metric_dict['epoch'],metric_dict['mse'],lr,
                                        metric_dict['r2'],
                                        metric_dict['pearsonr'],
                                        metric_dict['concordance_index'],
                                        metric_dict['explained_variance'],
                                        metric_dict['cindex'],
                                        metric_dict['rm2'],
                                        ))

            if metric_dict['prefix'] == 'tst' :
                print('\n%s:Ep%04d|| %.03f,%.03f,%.03f,%.03f,%.03f,%.03f,%.03f'%(
                                        metric_dict['prefix'],metric_dict['epoch'],metric_dict['mse'],
                                        metric_dict['r2'],
                                        metric_dict['pearsonr'],
                                        metric_dict['concordance_index'],
                                        metric_dict['explained_variance'],
                                        metric_dict['cindex'],
                                        metric_dict['rm2'],
                                        ))


    def cal_metrics_on_epoch_end(self,y_true,y_pred,prefix,current_epoch=None ):

        if self.task_type in ['multi_classification','multiclass',]:
            metric_dict =evaluate_multiclass(y_true,y_pred,to_categorical=True,num_classes=self.num_classes)     
        elif self.task_type in ['binary','binary_classification']:
            metric_dict =evaluate_binary(y_true,y_pred)
        elif self.task_type in ['multilabel_classification','multilabel',]:
            metric_dict = evaluate_multilabel(y_true,y_pred)
        elif self.task_type in ['regression',]:
            metric_dict = evaluate_regression(y_true,y_pred)    
        metric_dict['prefix'] = prefix
        metric_dict['epoch'] = self.current_epoch if current_epoch is None else current_epoch
        return metric_dict


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


    def configure_optimizers(self):

        self.my_optimizers =  torch.optim.Adam(self.parameters(), lr=self.lr)


        if self.scheduler_ReduceLROnPlateau_tracking in ['mse',]:
            mode = 'min'
        elif self.scheduler_ReduceLROnPlateau_tracking in ['F1','auPRC']:
            mode = 'max'
        else: raise 
        self.my_schedulers = t.optim.lr_scheduler.ReduceLROnPlateau(self.my_optimizers,
                                mode= mode,#'min',
                                factor=0.1, patience=8, verbose=True, 
                                threshold=0.0001, threshold_mode='abs', )
        return self.my_optimizers 

