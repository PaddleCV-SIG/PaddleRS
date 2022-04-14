import sys
import os
import cv2
import argparse
import paddlers as pdrs
import paddle
import numpy as np
import tqdm
from paddlers import transforms as T

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infer_dir',
        '-m',
        type=str,
        default=None,
        help='model directory path')
    parser.add_argument(
        '--img_dir',
        '-s',
        type=str,
        default=None,
        help='path to save inference model')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./tutorials/infer/output',
        help='path to save inference result')
    parser.add_argument(
        '--warmup_iters', type=int, default=0, help='warmup_iters')

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser.add_argument('--repeats', type=int, default=1, help='repeats')

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser


def quantize(arr):
    return (arr * 255).astype('uint8')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()


    eval_transforms = T.Compose([
        T.Resize(target_size=256),
        T.Normalize(
          mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    predictor = pdrs.deploy.Predictor(args.infer_dir, use_gpu=args.use_gpu)
    img_dir = args.img_dir
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    warmup_iters = args.warmup_iters
    repeats = args.repeats
    phase = 'test'
    prefix = '.png'
    floder1 = os.path.join(img_dir, phase, 'A')
    floder2 = os.path.join(img_dir, phase, 'B')

    img_list1 = [
        f for f in os.listdir(os.path.join(img_dir, phase, 'A'))
        if f.endswith(prefix)
    ]
    print('total file number is {}'.format(len(img_list1)))
    for filename in img_list1:
        imgfile = (os.path.join(floder1, filename), os.path.join(floder2,
                                                                 filename))
        result = predictor.predict(
            img_file=imgfile, warmup_iters=warmup_iters, repeats=repeats,transforms=eval_transforms)
        #label_map = result[0]['label_map']
        score_map = result[0]['score_map'][:, :, -1]
        cv2.imwrite(
            os.path.join(output_dir, filename), quantize(score_map > 0.5))
