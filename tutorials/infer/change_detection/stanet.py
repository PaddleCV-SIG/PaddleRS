import sys
import os
import cv2
import argparse
import paddlers as pdrs
import paddle
import numpy as np
import tqdm
from paddlers import transforms as T


def quantize(arr):
    return (arr * 255).astype('uint8')


infer_dir = "./inference_model"
img_dir = "./STANET_Paddle/test_tipc/data/mini_levir_dataset/"
output_dir = "./STANET_Paddle/test_tipc/result/predict_output"
warmup_iters = 0
repeats = 1
phase = 'test'
prefix = '.png'

eval_transforms = T.Compose([
    T.Resize(target_size=256),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
predictor = pdrs.deploy.Predictor(infer_dir, use_gpu=True)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

floder1 = os.path.join(img_dir, phase, 'A')
floder2 = os.path.join(img_dir, phase, 'B')

img_list1 = [
    f for f in os.listdir(os.path.join(img_dir, phase, 'A'))
    if f.endswith(prefix)
]
print('total file number is {}'.format(len(img_list1)))
for filename in img_list1:
    imgfile = (os.path.join(floder1, filename), os.path.join(floder2, filename))
    result = predictor.predict(
        img_file=imgfile,
        warmup_iters=warmup_iters,
        repeats=repeats,
        transforms=eval_transforms)
    #label_map = result[0]['label_map']
    score_map = result[0]['score_map'][:, :, -1]
    cv2.imwrite(os.path.join(output_dir, filename), quantize(score_map > 0.5))
