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

import inspect
from contextlib import contextmanager
from functools import wraps

import paddlers.transforms.operators as op
from .operators import *
from .batch_operators import BatchRandomResize, BatchRandomResizeByShort, _BatchPadding


def _get_arrange_transform(model_type, mode):
    apply_im_only = None
    if model_type == 'segmenter':
        if mode == 'eval':
            apply_im_only = True
        else:
            apply_im_only = False
        arrange_transform = op.ArrangeSegmenter(mode)
    elif model_type == 'changedetector':
        if mode == 'eval':
            apply_im_only = True
        else:
            apply_im_only = False
        arrange_transform = op.ArrangeChangeDetector(mode)
    elif model_type == 'classifier':
        arrange_transform = op.ArrangeClassifier(mode)
    elif model_type == 'detector':
        arrange_transform = op.ArrangeDetector(mode)
    else:
        raise Exception("Unrecognized model type: {}".format(model_type))
    return arrange_transform, apply_im_only


def arrange_transforms(model_type, transforms, mode='train'):
    # 给transforms添加arrange操作
    arrange_transform, apply_im_only = _get_arrange_transform(model_type, mode)
    transforms.arrange_outputs = arrange_transform
    if apply_im_only is not None:
        transforms.apply_im_only = apply_im_only


@contextmanager
def arrange_transforms_ctx(model_type, transforms, mode='train'):
    old_arrange_transform = transforms.arrange_outputs
    old_apply_im_only = transforms.apply_im_only
    arrange_transform, apply_im_only = _get_arrange_transform(model_type, mode)
    try:
        transforms.arrange_outputs = arrange_transform
        if apply_im_only is not None:
            transforms.apply_im_only = apply_im_only
        yield transforms
    finally:
        transforms.arrange_outputs = old_arrange_transform
        transforms.apply_im_only = old_apply_im_only


def arrange_transforms_deco(param_name, model_type='auto', mode='train'):
    def _deco(func):
        @wraps(func)
        def _wrapper(self, *args, **kwargs):
            nonlocal model_type
            if model_type == 'auto':
                model_type = self.model_type
            call_args = inspect.getcallargs(func, self, *args, **kwargs)
            arg = call_args[param_name]
            if not isinstance(arg, Compose):
                arg = arg.transforms
            with arrange_transforms_ctx(model_type, arg, mode):
                return func(self, *args, **kwargs)

        return _wrapper

    return _deco


def build_transforms(transforms_info):
    import paddlers.transforms as T
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise Exception("There's no transform named '{}'".format(op_name))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    return eval_transforms
