# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import argparse
from math import ceil
from PIL import Image
import numpy as np


# from utils import Raster


# def file_name(file_dir):
#     for root, dirs, files in os.walk(file_dir):
#         print('root_dir:', root)  # 当前目录路径
#         print('sub_dirs:', dirs)  # 当前路径下所有子目录
#         print('files:', files)  # 当前路径下所有非目录子文件


def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename);
    return shotname, extension


def create_list(data_dir, A, B, label, txt_save_path):
    img_ext = [".jpg", ".png"]
    file_list = []
    pth = osp.join(data_dir, A)
    for root, dirs, files in os.walk(pth):
        print(root)
        for file in files:
            shotname, extension = GetFileNameAndExt(file)
            if extension in img_ext:
                file_list.append(file)
    with open(txt_save_path, 'w') as f:
        for i in range(len(file_list)):
            f.write(osp.join(A, file_list[i]) + " " + osp.join(B, file_list[i]) + " " + osp.join(label,
                                                                                                 file_list[i]) + "\r")


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--image_folder", type=str, required=True, \
                    help="The path of original dataset.")
parser.add_argument("--A", type=str, required=True, \
                    help="The path of T1 image folder.")
parser.add_argument("--B", type=str, required=True, \
                    help="The path of T2 image folder")
parser.add_argument("--label", type=str, required=True,\
                    help="The path of label folder.")
parser.add_argument("--save_txt", type=str, required=True, \
                    help="The path to save the txt")

if __name__ == "__main__":
    args = parser.parse_args()
    create_list(args.image_folder, args.A,args.B,args.label, args.save_txt)
    # create_list("../dataset/train", "A", "B", "label", 'train.txt')
