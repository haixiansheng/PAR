import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import glob
from pathlib import Path
from tools.function import get_pkl_rootpath
import torchvision.transforms as T

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        # print(img_id)
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        # self.attr_num = len(self.attr_id)
        self.attr_num = 18

        self.img_idx = dataset_info.partition[split]
        # print(self.img_idx)
        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]
        print("111",len(self.label))
        #在之前的数据后面加上两列0，组成18个属性值
        n = np.zeros((len(self.label),2))
        self.label = np.hstack((self.label, n))
        # print(type(self.img_id))

        if split== args.train_split:
            for _, dirnames, images in os.walk("./data_hh"):
                for dirname in dirnames:
                    for file in os.listdir(os.path.join("./data_hh",dirname)):
                        self.img_id=np.append(self.img_id,[os.path.join(os.path.join("./data_hh",dirname),file)])
                        # imgs_path = os.path.join(os.path.join("./data_hh",dirname),file)
                        if "1-smoking" in dirname:
                            self.label=np.row_stack((self.label,np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])))
                            continue
                        elif "2-calling" in dirname:
                            self.label=np.row_stack((self.label,np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])))
                            continue
                        elif "3-calling&smoking" in dirname:
                            self.label=np.row_stack((self.label,np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]])))
                            continue
                        elif "4-others" in dirname:
                            self.label=np.row_stack((self.label,np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
                            continue
                        else:
                            pass
                            
        print(len(self.img_id))
        print(len(self.label))
        print(self.img_id[-1])
        print(self.label[-1])

        
    def __getitem__(self, index):
        if "data_hh/" not in self.img_id[index]: 
            imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
            imgpath = os.path.join(self.root_path, imgname)
            # img = Image.open(imgpath)       

        else:
            imgname, gt_label = self.img_id[index], self.label[index]
            imgpath = imgname
            # imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

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
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


class LoadImages:  # for inference

    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

