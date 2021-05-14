import os
import pprint
import math
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.zhikunData import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet101,resnet50,resnet18
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

set_seed(605)

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'zhikun_attr18_20210508.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split,transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_set = AttrDataset( args=args, split=args.valid_split,transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # print(f'{args.train_split} set: {len(train_loader.dataset)}, '
    #       f'{args.valid_split} set: {len(valid_loader.dataset)}, '
    #       f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    # backbone = resnet50(pretrained=True) #
    backbone = resnet18(pretrained=True)   ####使用resnet18需要修改base_block.py的nn.Linear(512, nattr)
    print("resnet18!")

    classifier = BaseClassifier(nattr=2)
    model = FeatClassifier(backbone, classifier)

    # state_dict = torch.load("weights/ckpt_hh.pth")['state_dicts']
    # model.load_state_dict(state_dict) 

    # ###pretrain####
    # state_dict = torch.load("weights/ckpt_max.pth")['state_dicts']
    # model.load_state_dict(state_dict) 
    # print("pretrain model") 
    # model.classifier.logits = nn.Sequential(
    #         nn.Linear(2048, 2),
    #         nn.BatchNorm1d(2)
    #     )


    if torch.cuda.is_available():   
        model = torch.nn.DataParallel(model).cuda().module
        print('-------------------use cuda----------------- ')

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
    best_epoch = 0
    epoch_train_loss = []
    epoch_valid_loss = []

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
        with open('./exp_result/PA100k/train_epoch_info.txt', 'a+') as wf:
                wf.writelines(str(train_loss) + '\n')
        with open('./exp_result/PA100k/valid_epoch_info.txt', 'a+') as wf:
                wf.writelines(str(valid_loss) + '\n')
        # lr_scheduler.step(metrics=valid_loss, epoch=i)
        lr_scheduler.step()

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

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    parser = argument_parser()
    args = parser.parse_args()
    print(args)
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
