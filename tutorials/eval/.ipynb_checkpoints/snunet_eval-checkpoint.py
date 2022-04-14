#!/usr/bin/env python

# 变化检测模型SNUNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('/home/aistudio/PaddleRS')
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--out_dir', '-s', type=str, default=None, help='path to save inference model')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--weight_path', type=str, default=r"./output/snunet/best_model/model.pdparams", help='weight path')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST_PATH = os.path.join(DATA_DIR,'train.txt')
    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR,'test.txt')
    EXP_DIR = args.out_dir
    #测试阶段 batch size
    EVAL_BATCH_SIZE = args.batch_size
    state_dict_path = args.weight_path

    train_transforms = T.Compose([
        # 以50%的概率实施随机水平翻转
        T.RandomHorizontalFlip(prob=0.5),
        # 以50%的概率实施随机垂直翻转
        T.RandomVerticalFlip(prob=0.5),
        T.Maintain()
    ])

    eval_transforms = T.Compose([
        T.Maintain()
        # 验证阶段与训练阶段的数据归一化方式必须相同
    ])

    # 分别构建训练和验证所用的数据集
    train_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR,
        file_list=TRAIN_FILE_LIST_PATH,
        label_list=None,
        transforms=train_transforms,
        num_workers=4,
        shuffle=True,
        with_seg_labels=False,
        binarize_labels=True)

    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR,
        file_list=EVAL_FILE_LIST_PATH,
        label_list=None,
        transforms=eval_transforms,
        num_workers=0,
        shuffle=False,
        with_seg_labels=False,
        binarize_labels=True)

    # 使用默认参数构建SNUNet模型
    model = pdrs.tasks.SNUNet()

    model.net_initialize(pretrain_weights = state_dict_path)
    eval_metrics = model.evaluate(eval_dataset)
    print(str(eval_metrics))

