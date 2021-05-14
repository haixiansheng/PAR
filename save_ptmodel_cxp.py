import numpy as np
import torch
import cv2
from PIL import Image
# from .dataset.AttrDataset import AttrDataset, get_transform, LoadImages
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier, module
from models.resnet import resnet50, resnet18
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
import argparse


def car_attr_init():
    # cfg = argument_parser().parse_args()
    attr_num = 16
    # device = cfg.device
    model_path = "exp_result/PA100k/PA100k/img_model/cxp_person_attr_resnet18_4cls.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(pretrained=False)
    classifier = BaseClassifier(nattr=4)
    model = FeatClassifier(backbone, classifier)
    # model = module(model)
    state_dict = torch.load(model_path, map_location=device)['state_dicts']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def model_transform(model):
    # model.cuda()
    torch.save(model.state_dict(), "cxp_person_attr_resnet18_4cls.pt", _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    model = car_attr_init()
    model_transform(model)
