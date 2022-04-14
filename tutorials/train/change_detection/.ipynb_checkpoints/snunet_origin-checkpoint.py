#!/usr/bin/env python

# 变化检测模型SNUNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('/home/aistudio/PaddleRS')
import paddle

import paddlers as pdrs
from paddlers import transforms as T

# 数据集存放目录
DATA_DIR = '../work/Real/subset'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '../work/Real/subset/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = '../work/Real/subset/test.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR = './output/snunet/'

# 下载和解压AirChange数据集
# airchange_dataset = 'http://mplab.sztaki.hu/~bcsaba/test/SZTAKI_AirChange_Benchmark.zip'
# pdrs.utils.download_and_decompress(airchange_dataset, path=DATA_DIR)

# 定义训练和验证时使用的数据变换（数据增强、预处理等）
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
# API说明：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/docs/apis/transforms.md
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
# 目前已支持的模型请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/docs/apis/model_zoo.md
# 模型输入参数请参考：https://github.com/PaddleCV-SIG/PaddleRS/blob/develop/paddlers/tasks/changedetector.py
model = pdrs.tasks.SNUNet()

# state_dict_path = r"output/snunet/best_model/model.pdparams"
# model.net_initialize(pretrain_weights = state_dict_path)

# 初始学习率
LR = 0.001
# 学习率衰减步长（注意，单位为迭代次数而非epoch数），即每多少次迭代将学习率衰减一半
DECAY_STEP = 5000
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
# 执行模型训练
model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    optimizer = optimizer,
    save_interval_epochs=5,
    # 每多少次迭代记录一次日志
    log_interval_steps=25,
    save_dir=EXP_DIR,
    # 是否使用early stopping策略，当精度不再改善时提前终止训练
    early_stop=False,
    # 是否启用VisualDL日志功能
    use_vdl=True,
    # 指定从某个检查点继续训练
    resume_checkpoint=None)
