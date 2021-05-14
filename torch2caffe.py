import torch
# from tools import torch_utils
from torchvision.models import resnet
from pytorch2caffe import pytorch2caffe
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet18
import os


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    name = 'car_attr18'

    # device = torch_utils.select_device('1')
    backbone = resnet18(pretrained=False)
    # backbone = resnet101()
    classifier = BaseClassifier(nattr=17)
    model = FeatClassifier(backbone, classifier)
    state_dict = torch.load('weights/car_attr18_20210512.pt', map_location="cpu")  # ['state_dicts']
    # state_dict = remove_prefix(state_dict, 'module.')
    print(state_dict.keys())

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.ones([1, 3, 224, 224])
    pytorch2caffe.trans_net(model, dummy_input, name)
    pytorch2caffe.save_prototxt('weights/{}.prototxt'.format(name))
    pytorch2caffe.save_caffemodel('weights/{}.caffemodel'.format(name))
