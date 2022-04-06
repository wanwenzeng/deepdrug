import os
import warnings
import numpy as np
import yaml
import pandas as pd
import yaml
warnings.filterwarnings('ignore')
import argparse
import torch as t 
from torch import Tensor
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from model import DeepDrug_Container
from dataset import DeepDrug_Dataset
from utils import * 

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--task", type=str, default='',help="task type.")
    # parser.add_argument("-d", "--dataset", type=str, default='', help="Dataset file path.")
    # parser.add_argument("-cv", "--cv_fold", type=int, default=0, help="cross validation fold.")
    # parser.add_argument('-sf','--save_folder',type=str,default='')
    # parser.add_argument('-g','--gpus',nargs='+',type=int,default=[0,])
    # parser.add_argument('-bs','--batch_size',type=int,default=256)
    # parser.add_argument('-e1_file','--entry1_file',type=str,default='')
    # parser.add_argument('-e2_file','--entry2_file',type=str,default='')
    # parser.add_argument('-e2_seq_file','--entry2_seq_file',type=str,default='None')
    # parser.add_argument('-e1_seq_file','--entry1_seq_file',type=str,default='None')
    # parser.add_argument('-p_file','--pair_file',type=str,default='')
    # parser.add_argument('-l_file','--label_file',type=str,default='')
    # parser.add_argument('-cv_file','--cv_file',type=str,default='')
    # parser.add_argument('-v','--version',type=str,default='')
    parser.add_argument('-cf','--configfile',type=str,default='')
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    configfile = args.configfile
    with open(configfile, 'r') as f:
        config = yaml.load(f)
        print(config)
    args = Struct(**config)
    

    entry1_data_folder = '/'.join(args.entry1_file.split('/')[:-2])
    entry2_data_folder = '/'.join(args.entry2_file.split('/')[:-2])
    entry2_seq_file = args.entry2_seq_file
    entry1_seq_file = args.entry1_seq_file
    assert os.path.exists(entry1_seq_file),'file does not exist: %s.'%entry1_seq_file 
    assert os.path.exists(entry2_seq_file),'file does not exist: %s.'%entry2_seq_file
    entry_pairs_file = args.pair_file
    pair_labels_file = args.label_file 
    save_folder = args.save_folder 
    dataset = args.dataset
    save_model_folder = pathjoin(save_folder,'models')
    y_true_file = pathjoin(save_folder,'test_true.csv')
    y_pred_file = pathjoin(save_folder,'test_pred.csv')
    os.makedirs(save_folder,exist_ok=True)
    os.makedirs(pathjoin(save_folder,'plots'),exist_ok=True)
    task_type = args.task
    dataset = args.dataset 
    gpus = args.gpus 
    category  = args.category 
    num_out_dim = args.num_out_dim
    



    y_transfrom_func = None
    if (dataset in ['DAVIS','BindingDB']) and (task_type =='regression') :
            y_transfrom_func = y_log10_transfrom_func

    if args.task in ['binary','multiclass','multilabel']:
        scheduler_ReduceLROnPlateau_tracking = 'F1'
        earlystopping_tracking = "val_epoch_F1"
    else:
        earlystopping_tracking='val_loss'
        scheduler_ReduceLROnPlateau_tracking= 'mse'

    kwargs_dict = dict(save_folder=save_folder,task_type=task_type,
            gpus=gpus,
            entry1_data_folder=entry1_data_folder,
            entry2_data_folder=entry2_data_folder,entry_pairs_file=entry_pairs_file,
            pair_labels_file=pair_labels_file,
            entry1_seq_file =entry1_seq_file ,entry2_seq_file = entry2_seq_file,
            y_true_file=y_true_file,y_pred_file=y_pred_file,
            y_transfrom_func=y_transfrom_func,
            earlystopping_tracking=earlystopping_tracking,
            scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,    
            )
            


    ######  for original training ##########
    _ = print_args(**kwargs_dict)

    datamodule = DeepDrug_Dataset(entry1_data_folder,entry2_data_folder,entry_pairs_file,
                        pair_labels_file,
                        task_type = task_type,
                        y_transfrom_func=y_transfrom_func,
                        entry2_seq_file = entry2_seq_file,
                        entry1_seq_file = entry1_seq_file,
                        category=category,
                        )

    model =  DeepDrug_Container(
                            task_type = task_type,category=category,
                            scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,
                            num_out_dim = num_out_dim,
                            )

    if earlystopping_tracking in ['val_loss',]:
        earlystopping_tracking = earlystopping_tracking
        earlystopping_mode = 'min'
        earlystopping_min_delta = 0.0001
    elif earlystopping_tracking in ['val_epoch_F1','val_epoch_auPRC']:
        earlystopping_mode = 'max'
        earlystopping_min_delta = 0.001
    else:
        raise 
    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=save_model_folder,
                                        mode = earlystopping_mode,
                                        monitor=earlystopping_tracking,
                                        save_top_k=1,save_last=True,)
    earlystop_callback = pl_callbacks.EarlyStopping(earlystopping_tracking,verbose=True,
                                        mode = earlystopping_mode,
                                        min_delta=earlystopping_min_delta,
                                        patience=10,)
    
    trainer = Trainer(
                    gpus=[gpus,],
                    accelerator=None,
                    max_epochs=200, min_epochs=5,
                    default_root_dir= save_folder,
                    fast_dev_run=False,
                    check_val_every_n_epoch=1,
                    callbacks=  [checkpoint_callback,
                                earlystop_callback,],
                    )
    trainer.fit(model, datamodule=datamodule,)


    ################  Prediction ##################
    print('loading best weight in %s ...'%(checkpoint_callback.best_model_path))
    model = model.load_from_checkpoint(checkpoint_callback.best_model_path,verbose=True)
    model.eval()    
    
    trainer.test(model,datamodule=datamodule,)
    y_pred = trainer.predict(model,dataloaders =datamodule.test_dataloader())
    y_true = np.array(datamodule.pair_labels[datamodule.test_indexs])

    if isinstance(y_pred[0],t.Tensor):
        y_pred = [x.cpu().data.numpy() for x in y_pred]
    if isinstance(y_pred,t.Tensor):
        y_pred = y_pred.cpu().data.numpy()
    y_pred = np.concatenate(y_pred,axis=0) 
    pd.DataFrame(y_pred).to_csv(y_pred_file,header=True,index=False)
    pd.DataFrame(y_true).to_csv(y_true_file,header=True,index=False)
    print('save prediction completed.')
