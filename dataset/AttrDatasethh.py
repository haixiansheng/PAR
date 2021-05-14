import os
import pickle
import random
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

    def __init__(self,transform=None, target_transform=None):
        # self.test=test
        # self.train=train
        self.transform=transform
        self.target_transform = target_transform
        # imgs=[os.path.join(root,os.path.join(dirname,img)) for _,dirname,img in os.walk(root)]
        imgs = []
        self.label = []
        for _, dirnames, images in os.walk("./data_hh"):
            for dirname in dirnames:
                for file in os.listdir(os.path.join("./data_hh",dirname)):
                    imgs.append(os.path.join(os.path.join("./data_hh",dirname),file))
                    # imgs_path = os.path.join(os.path.join("./data_hh",dirname),file)
                    if "1-smoking" in dirname:
                        self.label.append([1,0,0])
                    elif "2-calling" in dirname:
                        self.label.append([0,1,0])
                    elif "3-hold_gun" in dirname:
                        self.label.append([0,0,1])
                    elif "4-others" in dirname:
                        self.label.append([0,0,0])
                    else:
                        pass
            # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg

        # if self.test:
        # 	imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        # else:
        # 	imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2]))
        self.label = np.asarray(self.label)
        self.attr_num = 3
        imgs_num=len(imgs)
        self.imgs=imgs
        print("imgs_len:",len(self.imgs))
        print("label_len:",len(self.label))
        # if self.test:
        #     self.imgs=imgs
        # else:
        #     random.shuffle(imgs)
        #     if self.train:
        #         self.imgs=imgs
        #     else:
        #         self.imgs=imgs

    def __getitem__(self, index):

        imgname, gt_label = self.imgs[index], self.label[index]
        # imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgname)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.imgs)


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

