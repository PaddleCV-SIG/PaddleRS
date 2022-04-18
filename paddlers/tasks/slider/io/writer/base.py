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
from typing import Dict


class BaseWriter(object):
    def __init__(self, config: Dict, ext: str="tif") -> None:
        file_split = osp.split(config["file_path"])
        self.save_path = osp.join(
            file_split[0], osp.splitext(file_split[-1])[0] + "_output." + ext)
        self.width = config["width"]
        self.height = config["height"]
        self.proj = config["proj"]
        self.geotf = config["geotf"]
        self.block_size = config["block_size"]
        self.overlap = config["overlap"]
        self.num_block = config["num_block"]

    def write(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
