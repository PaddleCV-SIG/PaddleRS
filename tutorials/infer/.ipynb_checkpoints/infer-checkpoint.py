import sys
sys.path.append('../PaddleRS')
import os
import cv2
import argparse
import paddlers as pdrs
import paddle
import numpy as np
import tqdm





def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--img_dir', '-s', type=str, default=None, help='path to save inference model')
    parser.add_argument('--output_dir', type=str, default='./tutorials/infer/output', help='path to save inference result')
    parser.add_argument('--warmup_iters', type=int, default=0, help='warmup_iters')
    parser.add_argument('--repeats', type=int, default=1, help='repeats')

    return parser

def quantize(arr):
    return (arr*255).astype('uint8')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    predictor = pdrs.deploy.Predictor(args.infer_dir, use_gpu=True)
    img_dir = args.img_dir
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    warmup_iters=args.warmup_iters
    repeats=args.repeats
    phase = 'test'
    prefix = '.jpg'
    floder1 = os.path.join(img_dir, phase, 'A')
    floder2 = os.path.join(img_dir, phase, 'B')

    img_list1 = [f for f in os.listdir(os.path.join(img_dir, phase, 'A')) if f.endswith(prefix)]
    print('total file number is {}'.format(len(img_list1)))
    for filename in img_list1:
        imgfile = (os.path.join(floder1, filename), os.path.join(floder2, filename))
        result = predictor.predict(img_file=imgfile, warmup_iters=warmup_iters, repeats=repeats)
        #label_map = result[0]['label_map']
        score_map = result[0]['score_map'][:, :, -1]
        cv2.imwrite(os.path.join(output_dir, filename), quantize(score_map>0.5))