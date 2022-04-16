#!/usr/bin/env python

# 变化检测模型SNUNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('../PaddleRS')
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument(
        '--out_dir',
        '-s',
        type=str,
        default=None,
        help='path to save inference model')
    parser.add_argument('-lr', type=float, default=0.01, help='lr')
    parser.add_argument(
        '--num_epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument(
        '--train_num', type=int, default=7120, help='train dataset num')
    parser.add_argument('--save_epoch', type=int, default=5, help='save epoch')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST_PATH = os.path.join(DATA_DIR, 'train.txt')
    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR, 'val.txt')
    EXP_DIR = args.out_dir
    LR = args.lr
    NUM_EPOCHS = args.num_epoch
    # 每多少个epoch保存一次模型权重参数
    SAVE_INTERVAL_EPOCHS = args.save_epoch
    #训练阶段 batch size
    TRAIN_BATCH_SIZE = args.batch_size
    MAX_STEP = int(NUM_EPOCHS * (args.train_num / TRAIN_BATCH_SIZE))
    print("MAX STEP IS {}".format(MAX_STEP))

    train_transforms = T.Compose([
        # 随机裁剪
        T.RandomCrop(
            # 裁剪区域将被缩放到256x256
            crop_size=256,
            # 裁剪区域的横纵比在0.5-2之间变动
            aspect_ratio=[1.0, 1.0],
            # 裁剪区域相对原始影像长宽比例在一定范围内变动，最小不低于原始长宽的4/5
            scaling=[0.8, 1.0]),
        # 以50%的概率实施随机水平翻转
        T.RandomHorizontalFlip(prob=0.5),
        T.RandomVerticalFlip(prob=0.5),
        T.RandomBlur(),
        # 将数据归一化到[-1,1]
        T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transforms = T.Compose([
        # 验证阶段与训练阶段的数据归一化方式必须相同
        T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    # 目前已支持的模型请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/docs/apis/model_zoo.md
    # 模型输入参数请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/paddlers/tasks/changedetector.py
    model = pdrs.tasks.BIT()
    # 线性学习率
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        learning_rate=LR,
        lr_lambda=lambda x: 1 - x / float(NUM_EPOCHS + 1),
        verbose=False)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model.net.parameters(),
        momentum=0.9,
        weight_decay=5e-4)

    # 执行模型训练
    model.train(
        num_epochs=NUM_EPOCHS,
        train_dataset=train_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        save_interval_epochs=SAVE_INTERVAL_EPOCHS,
        # 每多少次迭代记录一次日志
        log_interval_steps=20,
        save_dir=EXP_DIR,
        # 是否使用early stopping策略，当精度不再改善时提前终止训练
        early_stop=False,
        # 是否启用VisualDL日志功能
        use_vdl=True,
        # 指定从某个检查点继续训练
        resume_checkpoint=None)
