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

import os.path as osp
import copy

from paddle.io import Dataset
from paddlers.utils import logging, get_num_workers, get_encoding, path_normalization, is_pic


class ClasDataset(Dataset):
    """读取图像分类任务数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注序号（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径，文件格式为（类别 说明）。默认值为None。
        transforms (paddlers.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list=None,
                 transforms=None,
                 num_workers='auto',
                 shuffle=False):
        super(ClasDataset, self).__init__()
        self.transforms = copy.deepcopy(transforms)
        # TODO batch padding
        self.batch_transforms = None
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        self.file_list = list()
        self.label_list = label_list
        self.labels = list()

        # TODO：非None时，让用户跳转数据集分析生成label_list
        # 不要在此处分析label file
        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)
        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 2:
                    raise Exception(
                        "A space is defined as the delimiter to separate the image and label path, " \
                        "so the space cannot be in the image or label path, but the line[{}] of " \
                        " file_list[{}] has a space in the image or label path.".format(line, file_list))
                items[0] = path_normalization(items[0])
                full_path_im = osp.join(data_dir, items[0])
                label = items[1]
                if not is_pic(full_path_im):
                    continue
                if not osp.exists(full_path_im):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im))
                if not label.isdigit():
                    raise ValueError(
                        'Label {} does not convert to number(int)!'.format(
                            label))
                self.file_list.append({
                    'image': full_path_im,
                    'label': int(label)
                })
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.file_list[idx])
        outputs = self.transforms(sample)
        return outputs

    def __len__(self):
        return len(self.file_list)
