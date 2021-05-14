import os
import pickle

import numpy as np
import torch.utils.data as data
import cv2
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T


def cv2_letterbox_image(image, expected_size):
    '''
    自适应维持比例resize
    '''
    # print("old img shape:", image.shape)
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    # print("new_img shape:", new_img.shape)
    return new_img


class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        # img = Image.open(imgpath)

        img = cv2.imread(imgpath)  ### 会改变颜色空间
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2_letterbox_image(img, (224, 224))  # 维持训练图像长宽比为128，86
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        # T.RandomCrop((height, width)),
        T.ColorJitter(0.8, 0.8, 0.5),
        T.RandomHorizontalFlip(),
        T.RandomRotation(45),
        T.ToTensor(),
        normalize,
    ])
    # import torchvision.transforms as T
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
