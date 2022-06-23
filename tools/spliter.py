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

from utils import Raster, use_time


@use_time
def split_data(image_path, mask_path, block_size, save_folder):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(osp.join(save_folder, "images"))
        if mask_path is not None:
            os.makedirs(osp.join(save_folder, "masks"))
    image_name = image_path.replace("\\", "/").split("/")[-1].split(".")[0]
    image = Raster(image_path, to_uint8=True)
    mask = Raster(mask_path) if mask_path is not None else None
    if image.width != mask.width or image.height != mask.height:
        raise ValueError("image's shape must equal mask's shape.")
    rows = ceil(image.height / block_size)
    cols = ceil(image.width / block_size)
    total_number = int(rows * cols)
    for r in range(rows):
        for c in range(cols):
            loc_start = (c * block_size, r * block_size)
            image_title = Image.fromarray(image.getArray(
                loc_start, (block_size, block_size))).convert("RGB")
            image_save_path = osp.join(save_folder, "images", (
                image_name + "_" + str(r) + "_" + str(c) + ".jpg"))
            image_title.save(image_save_path, "JPEG")
            if mask is not None:
                mask_title = Image.fromarray(mask.getArray(
                    loc_start, (block_size, block_size))).convert("L")
                mask_save_path = osp.join(save_folder, "masks", (
                    image_name + "_" + str(r) + "_" + str(c) + ".png"))
                mask_title.save(mask_save_path, "PNG")
            print("-- {:d}/{:d} --".format(int(r * cols + c + 1), total_number))


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--image_path", type=str, required=True, \
                    help="The path of big image data.")
parser.add_argument("--mask_path", type=str, default=None, \
                    help="The path of big image label data.")
parser.add_argument("--block_size", type=int, default=512, \
                    help="The size of image block, `512` is the default.")
parser.add_argument("--save_folder", type=str, default="dataset", \
                    help="The folder path to save the results, `dataset` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    split_data(args.image_path, args.mask_path, args.block_size, args.save_folder)
