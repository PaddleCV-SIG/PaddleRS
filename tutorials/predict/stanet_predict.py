import sys
import paddle
import os
import argparse
#加入环境
sys.path.append('./STANET_Paddle/')
import paddlers as pdrs
from paddlers import transforms as T
import paddle.nn as nn
import paddle
from paddlers.transforms import arrange_transforms
from paddlers.transforms import ImgDecoder, Resize
import numpy as np
import  cv2
from collections import OrderedDict
from paddlers.custom_models.cd  import STANet

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default=None, help='path to save result url')
    parser.add_argument('--state_dict_path', type=str, default=None ,help='where model params')
    parser.add_argument('--img1', type=str, default=None, help='img1 url')
    parser.add_argument('--img2', type=str, default=None ,help='img2 url')
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    EXP_DIR = args.out_dir
    state_dict_path = args.state_dict_path
    num_classes = 2
    model = pdrs.tasks.STANet( in_channels=3, num_classes=num_classes, att_type='PAM', ds_factor=1)     
    model.net_initialize(pretrain_weights = state_dict_path)
    model.net.eval()
    eval_transforms = T.Compose([
        T.Resize(target_size=256),
        T.Normalize(
          mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = {'image_t1': args.img1, 'image_t2': args.img2}
    image = eval_transforms(image)
    t1 = image["image"].transpose((2, 0, 1))
    t2 = image["image2"].transpose((2, 0, 1))
    t1 = paddle.to_tensor(t1).unsqueeze(0)
    t2 = paddle.to_tensor(t2).unsqueeze(0)
    vis = paddle.argmax(model.net(t1, t2)[-1], 1)[0].numpy()
    vis = vis.astype("uint8")*255
    cv2.imwrite(EXP_DIR+"/result.png", vis)
    print('finish!')


