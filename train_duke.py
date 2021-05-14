#   /*coding: utf-8 */

import pprint
import math
from collections import OrderedDict, defaultdict
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from datafolder.folder import Train_Dataset




######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}


######################################################################
# Argument
# --------
def arg_parser():
    # 数据设置
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data-path', default='./data', type=str, help='path to the dataset')
    parser.add_argument('--dataset', default='duke', type=str, help='dataset: market, duke')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--num-workers', default=4, type=int, help='num_workers')
    parser.add_argument('--use-id', action='store_true', help='use identity loss')
    parser.add_argument('--lamba', default=1.0, type=float, help='weight of id loss')
    # 模型参数设置
    parser.add_argument("--debug", action='store_false')
    parser.add_argument("--train_epoch", type=int, default=120)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default='7', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')
    args = parser.parse_args()
    return args

######################################################################
# DataLoader
# ---------


def data_Loader(args):
    assert args.dataset in ['market', 'duke']
    dataset_name = dataset_dict[args.dataset]
    data_dir = args.data_path

    image_datasets = {}
    image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='train')
    image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='query')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers, drop_last=True)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # images, indices, labels, ids, cams, names = next(iter(dataloaders['train']))

    num_label = image_datasets['train'].num_label()
    num_id = image_datasets['train'].num_id()
    labels_list = image_datasets['train'].labels()
    train_data = dataloaders['train']
    valid_data = dataloaders['val']
    labels_weights = image_datasets['train'].distribution
    # print("1111111111",labels_weights)
    valid_labels = image_datasets['val'].distribution
    return train_data, valid_data, num_label, labels_weights, valid_labels


def main(args):
    visenv_name = args.dataset
    exp_dir = './exp_result'
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name) # 需要修改
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_loader, valid_loader, num_labels, train_labels, labels_weights = data_Loader(args)
    sample_weight = labels_weights
    backbone = resnet50()
    classifier = BaseClassifier(nattr=num_labels)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda().module

    criterion = CEL_Sigmoid(sample_weight)

    param_groups = [{'params': model.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lf = lambda x: (((1 + math.cos(x * math.pi / args.train_epoch)) / 2) ** 1.0) * 0.8 + 0.00000001  # cosine
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
    lr_scheduler.last_epoch = - 1
    # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    epoch_train_loss = []
    epoch_valid_loss = []
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )
        epoch_train_loss.append(train_loss)
        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        epoch_valid_loss.append(valid_loss)
        lr_scheduler.step()
        # lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum or i % 20 == 0:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    with open('./exp_result/duke/train_epoch_info.txt', 'w+') as wf:
        for info in epoch_train_loss:
            wf.writelines(str(info) + '\n')
    with open('./exp_result/duke/valid_epoch_info.txt', 'w+') as wf:
        for info in epoch_valid_loss:
            wf.writelines(str(info) + '\n')
    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))
    return maximum, best_epoch


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
    print('fdf')
    torch.cuda.empty_cache()
    args = arg_parser()
    main(args)
