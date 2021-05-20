'''
仅适用于rknn属性代码评测
'''
from easydict import EasyDict
import os
import numpy as np
from rknn.api import RKNN
import cv2

def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5):
    pred_label = preds_probs > threshold
    gt_label = gt_label > 0.8

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rknn_load_model(model_file_path):
    rknn = RKNN()
    print("-" * 100)
    print(f'--> Loading {model_file_path} model')
    ret = rknn.load_rknn(model_file_path)
    if ret != 0:
        print(f'load {model_file_path} rknn model failed')
        exit(ret)
    print(f'load {model_file_path} done')
    print(f'--> Init {model_file_path} runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print(f'Init {model_file_path}runtime environment failed')
        exit(ret)
    print(f'{model_file_path} init runtime done')
    return rknn


def rknn_inference(rknn,image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize_img = cv2.resize(image,(224,224))
    outputs = rknn.inference(inputs=[resize_img])[0]
    outputs = sigmoid(outputs)
    return outputs


def get_gt_label_and_predict_result(label_file_path):
    # 1.初始化模型
    model_path = "resnet18_attr.rknn"
    rknn = rknn_load_model(model_path)
    with open(label_file_path, "r") as f:
        lines = f.readlines()
        gt_lalel = []
        predict_result = []
        for line in lines:
            gt_label_line = []
            img_path = line.split(" ")[0]  # 文件名
            gt_label_str = line.split(" ")[1:]  # 真实标签
            if os.path.exists(img_path):
                # try:

                # print(label)
                for x in gt_label_str:
                    if x != "\n":
                        gt_label_line.append(int(x))
                    else:
                        continue

                if len(gt_label_line) == 16:
                    gt_label_line.append(0)  # 如果只有16位标签往最后填加一个0
                label_line = np.array(gt_label_line)
                ret = rknn_inference(rknn,img_path)
                ret = np.array(ret)

                predict_result.append(ret)
                gt_lalel.append(label_line)

            else:
                continue
            predict_result = np.array(predict_result)
            gt_lalel = np.array(gt_lalel)
            return gt_lalel, predict_result

if __name__ == "__main__":
    label_file_name = "data/val.txt"
    gt_lalel, predict_result = get_gt_label_and_predict_result(label_file_name)
    valid_result = get_pedestrian_metrics(gt_lalel, predict_result)
    print(f'Evaluation on test set, \n',
          'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
              valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
          'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
              valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
              valid_result.instance_f1))