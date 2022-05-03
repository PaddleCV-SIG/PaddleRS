import sys
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T
import paddle.nn as nn
import paddle

DATA_DIR = "./dataset/"
EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR, 'val.txt')

STATE_DICT_PATH = "./output/home/aistudio/output/stanet/best_model/model.pdparams"

eval_transforms = T.Compose([
    T.Resize(target_size=256),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
eval_dataset = pdrs.datasets.CDDataset(
    data_dir=DATA_DIR + '/val',
    file_list=EVAL_FILE_LIST_PATH,
    label_list=None,
    transforms=eval_transforms,
    num_workers=0,
    binarize_labels=True,
    shuffle=False)
# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/paddlers/blob/develop/docs/visualdl.md
num_classes = 2
model = pdrs.tasks.STANet(
    in_channels=3, num_classes=num_classes, att_type='PAM', ds_factor=1)
model.net_initialize(pretrain_weights=STATE_DICT_PATH)
model.net.eval()
eval_metrics = model.evaluate(eval_dataset)
print(str(eval_metrics))
