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
import paddle
from paddle.nn import Layer
from paddlers.transforms.operators import Transform
from tqdm import tqdm
from typing import List, Union
from .baseinfer import BaseSlider
from ..io import RasterLoader, SegWriter


class SegSlider(BaseSlider):
    def __init__(self, model: Layer,
                 transforms: Union[Transform, None]=None) -> None:
        """ Slide infer about segmentation.

        Args:
            model (Layer): Model of PaddleRS.
            transforms (Transform or None, optional): PaddleRS's transform. Defaults to None.
        """
        super(SegSlider, self).__init__(model, transforms)

    def __call__(self, path: str) -> None:
        dataloader = RasterLoader(path, self.block_size, self.overlap,
                                  self.transforms)
        datawriter = SegWriter(dataloader.config)
        for data in tqdm(dataloader):
            img = data["block"]
            start = data["start"]
            img = paddle.to_tensor(img.transpose((2, 0, 1))[None])
            with paddle.no_grad():
                pred = self.model(img)[0]
            block = paddle.argmax(
                pred, axis=1).squeeze().numpy().astype("uint8")
            datawriter.write(block, start)
        datawriter.close()
        print("[Finshed] The file saved {0}.".format(
            osp.normpath(datawriter.save_path)))

    def ready(self,
              block_size: Union[List[int], int]=512,
              overlap: Union[List[int], int]=32) -> None:
        """ Ready.

        Args:
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 512.
            overlap (Union[List[int], int], optional): Overlap between two images. Defaults to 32.
        """
        super(SegSlider, self).ready(block_size)
        self.overlap = overlap
