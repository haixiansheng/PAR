import os
import pickle
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile

import cv2
import glob
from pathlib import Path
from tools.function import get_pkl_rootpath
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']


def cv2_letterbox_image(image, expected_size):
    '''
    自适应维持比例resize
    '''
    print("old img shape:", image.shape)
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
    print("new_img shape:", new_img.shape)
    return new_img


class AttrDataset(data.Dataset):
    def __init__(self, split, args, transform=None, target_transform=None):
        # self.test=test
        # train = True
        self.transform = transform
        self.target_transform = target_transform
        # imgs=[os.path.join(root,os.path.join(dirname,img)) for _,dirname,img in os.walk(root)]
        imgs = []
        self.label = []
        if split == args.train_split:
            f = open("zhikun_attr/train.txt", 'r')
        elif split == args.valid_split:
            f = open("zhikun_attr/train.txt", 'r')
        else:
            print("no transform")
        lines = f.readlines()  # 读取整个文件所有行，保存在 list 列表中
        for line in lines:
            label_list = []
            img = line.split(" ")[0]  # 文件名
            label_str = line.split(" ")[1:]  # 标签
            if os.path.exists(img):
                # try:
                    
                    imgs.append(img)                
                    # print(label)
                    for x in label_str:
                        if x != "\n":
                            label_list.append(int(x))
                        else:
                            continue
                    self.label.append(label_list)
                # except:
                #     print(img)
            else:
                continue
            # print(x)

        self.imgs=imgs
        self.label = np.asarray(self.label)
        print(self.label[1])
        self.attr_num = 16
        imgs_num = len(imgs)
        # self.imgs=imgs
        print("imgs_len:", len(self.imgs))
        print("label_len:", len(self.label))
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
        
        # img = cv2.imread(imgname)  ###会改变颜色空间
        # img = cv2_letterbox_image(img, (256,192))  #维持训练图像长宽比为128，86
        # img = Image.fromarray(np.uint8(img))
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

    UNIT_SIZE = 200  # 每张图片的宽度是固定的
    size = (100, UNIT_SIZE)
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

