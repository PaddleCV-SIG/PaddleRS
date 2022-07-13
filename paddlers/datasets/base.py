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

from copy import deepcopy
from functools import wraps

from paddle.io import Dataset

from paddlers.utils import get_num_workers
from paddlers import global_options


class BaseDataset(Dataset):
    def __init__(self, data_dir, label_list, transforms, num_workers, shuffle):
        super(BaseDataset, self).__init__()

        self.data_dir = data_dir
        self.label_list = label_list
        self.transforms = deepcopy(transforms)
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle

    def __new__(cls, *unused_args, **unused_kwargs):
        # Do a late binding
        def _add_arrange_check(method):
            @wraps(method)
            def _wrapper(self, *args, **kwargs):
                if not global_options.ALLOW_NO_ARRANGE and not self.transforms.has_arrange:
                    raise RuntimeError(
                        "The output of transform operators has not been arranged. Please check if indexing of the dataset happened during model training, evaluation,"
                        "or inference. Please set `paddlrs.global_options.ALLOW_NO_ARRANGE` to True if your intention is to check the content of the dataset."
                    )
                return method(self, *args, **kwargs)

            return _wrapper

        cls.__getitem__ = _add_arrange_check(cls.__getitem__)
        return super().__new__(cls)
