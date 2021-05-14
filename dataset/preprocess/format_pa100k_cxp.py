import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

'''
pa100k_classes = ['帽子', '眼镜',
                  '短袖', '长袖', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
                  'LowerStripe', 'LowerPattern', '长外套', '长裤', '短裤', '长/短裙',
                  'boots',
                  '手提包', '肩包', '背包', 'HoldObjectsInFront',
                  'AgeOver60', 'Age18-60', 'AgeLess18',
                  'Female',
                  '正面', '侧面', '背面']

pa100k_classes = ['Hat', 'Glasses',
                  'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
                  'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress',
                  'boots',
                  'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
                  'AgeOver60', 'Age18-60', 'AgeLess18',
                  'Female',
                  'Front', 'Side', 'Back']

group_order = [7, 8,
               13, 14, 15, 16, 17, 18,
               19, 20, 21, 22, 23, 24,
               25,
               9, 10, 11, 12,
               1, 2, 3,
               0,
               4, 5, 6]
'''


pa100k_classes = ['Hat', 'Glasses',
                  'ShortSleeve', 'LongSleeve']

group_order = [0, 1,
               2, 3]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name

    # 只需要0-性别 1 2 3-年龄范围 7-帽子 8-眼镜 13-短袖 14-长袖 共8类
    targets = [7, 8, 13, 14]
    dataset.label = np.concatenate((pa100k_data['train_label'],
                                    pa100k_data['val_label'],
                                    pa100k_data['test_label']), axis=0)[:, targets]
    assert dataset.label.shape == (100000, 4)

    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in targets]

    # dataset.label = np.concatenate((pa100k_data['train_label'],
    #                                 pa100k_data['val_label'],
    #                                 pa100k_data['test_label']), axis=0)
    # assert dataset.label.shape == (100000, 26)
    # dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]

    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset_now.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":

    save_dir = './data/PA100k/'
    reoder = True
    generate_data_description(save_dir, reorder=True)
