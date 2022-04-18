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

from paddle.nn import Layer
from paddlers.transforms.operators import Transform
from typing import List, Union


class BaseSlider(object):
    def __init__(self, model: Layer,
                 transforms: Union[Transform, None]=None) -> None:
        self.model = model
        self.model.eval()
        self.transforms = transforms
        self.ready()

    def __call__(self) -> None:
        raise NotImplementedError()

    def ready(self, block_size: Union[List[int], int]=512) -> None:
        self.block_size = block_size
