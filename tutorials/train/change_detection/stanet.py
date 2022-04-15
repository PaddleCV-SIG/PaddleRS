#!/usr/bin/env python

# 变化检测模型STANet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库,并预处理数据集
import sys
sys.path.append('/home/aistudio/PaddleRS')
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T
import paddle.nn as nn
import paddle
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--out_dir', '-s', type=str, default=None, help='path to save inference model')
    parser.add_argument('-lr', type=float, default=0.001, help='lr')
    parser.add_argument('--decay_step', type=int, default=5000elp='epoch number')  
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=8 help='batch size')
    parser.add_argument('--save_epoch', type=int, default=3 help='save epoch')

    return parser

# # 数据集存放目录
# DATA_DIR = '../work/Real/subset'
# # 训练集`file_list`文件路径
# TRAIN_FILE_LIST_PATH = '../work/Real/subset/train.txt'
# # 验证集`file_list`文件路径
# EVAL_FILE_LIST_PATH = '../work/Real/subset/test.txt'
# # 实验目录，保存输出的模型权重和结果
# EXP_DIR = './output/snunet/'
# # 初始学习率
# LR = 0.001
# # 学习率衰减步长（注意，单位为迭代次数而非epoch数），即每多少次迭代将学习率衰减一半
# DECAY_STEP = 5000
# # 制定定步长学习率衰减策略

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    
    TRAIN_FILE_LIST_PATH = os.path.join(DATA_DIR,'train.txt')
    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR,'val.txt')
    TESTLE_LIST_PATH = os.path.join(DATA_DIR,'test.txt')
    LABEL_LIST_PATH = os.path.join(DATA_DIR,'label.txt
                                   
                                   
    
    
    EXP_DIR = args.out_dir
    
    LR = args.lr
    
    DECAY_STEP = args.decay_step
    NUM_EPOCHS = args.num_epoch
    # 每多少个epoch保存一次模型权重参数
    SAVE_INTERVAL_EPOCHS = args.save_epoch
    #训练阶段 batch size
    TRAIN_BATCH_SIZE = args.batch_size
    
    
    
    # 定义训练和验证时的transforms
    # API说明：https://github.com/PaddlePaddle/paddlers/blob/develop/docs/apis/transforms/transforms.md
    train_transforms = T.Compose([
        T.Resize(target_size=256),
        T.RandomHorizontalFlip(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transforms = T.Compose([
        T.Resize(target_size=256),
        T.Normalize(
          mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 定义训练和验证所用的数据集
    # API说明：https://github.com/PaddlePaddle/paddlers/blob/develop/docs/apis/datasets.md
    train_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR+'/train',
        file_list=TRAIN_FILE_LIST_PATH,
        label_list=LABEL_LIST_PATH,
        transforms=train_transforms,
        num_workers=2,
        binarize_labels=True,
        shuffle=True)

    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR+'/val',
        file_list=EVAL_FILE_LIST_PATH,
        label_list=LABEL_LIST_PATH,
        transforms=eval_transforms,
        num_workers=2,
        binarize_labels=True,
    shuffle=False)

    # 初始化模型，并进行训练
    # 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/paddlers/blob/develop/docs/visualdl.md
    num_classes = len(train_dataset.labels)


    model = pdrs.tasks.STANet( in_channels=3, num_classes=num_classes, att_type='PAM', ds_factor=1)



    # 制定定步长学习率衰减策略
    lr_scheduler = paddle.optimizer.lr.StepDecay(
        LR,
        step_size=DECAY_STEP,
        # 学习率衰减系数，这里指定每次减半
        gamma=0.5
    )

    # 构造Adam优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.net.parameters()
    )
    model.train(
        num_epochs=NUM_EPOCHS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        optimizer = optimizer,
        save_interval_epochs=SAVE_INTERVAL_EPOCHS,
        # 每多少次迭代记录一次日志
        log_interval_steps=20,
      
        # 是否使用early stopping策略，当精度不再改善时提前终止训练
        early_stop=False,
        # 是否启用VisualDL日志功能
        use_vdl=True,
        # pretrain_weights=None,
        save_dir='output/stanet'
        # 指定从某个检查点继续训练
        resume_checkpoint=None
        )
