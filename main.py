#coding=UTF-8
# 控制程序功能
# 保存数据


## multiDemo:
## python main.py -type multiDemo -database 2013 -sample 1_1 -israndom 1 -max_atoms 100

## train:
## python main.py -type train -sample 1_1 -max_atoms 50 -israndom 1 -database drugbank -location test -unit_list 128 128 128 128 128 128 128 128 -use_bn 1 -use_relu 1 -use_GMP 1 -gpu_id 1
import argparse
from run_deep_ddi import run
from train_deep_ddi import train
from demo import demo, multiDemo
# This for run
# python main.py -type run -max_atom 40 -sample 1_1 -gpu_id 1 -israndom 1 -database 2013
parser = argparse.ArgumentParser(description='Drug_Drug_Interaction main function')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, help='GPU devices')
parser.add_argument('-sample', dest='sample', type=str, help='1_1 or 1_2 or 1_4 or 1_8 or 1_16')
parser.add_argument('-type', dest='type', type=str, help='run or train or both or demo')
parser.add_argument('-max_atoms', dest='max_atoms', type=int, help='max atom of drug')
parser.add_argument('-israndom', dest='israndom', type=bool, help='randomlize X and A')
parser.add_argument('-database', dest='database', type=str, help="use which database, stanford, decagon, twosides, 2013")
parser.add_argument('-location', dest='location', type=str, help="locations of result under /DeepDDI_desktop/results. eg:test1/use_bn")

# This is for training
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')

args = parser.parse_args()


def main():
    if args.type == 'run':
        run(args.max_atoms, args.database, args.sample, args.israndom)
    elif args.type == 'train':
        train(args.location, args.max_atoms, args.gpu_id, args.database, \
            args.sample, args.unit_list, args.use_bn, \
                args.use_relu, args.use_GMP)
    elif args.type == 'demo':
        demo(args.max_atoms, args.israndom)
    elif args.type == 'multiDemo':
        multiDemo(args.database, args.sample, args.max_atoms, args.israndom)

if __name__ == "__main__":
    main()
