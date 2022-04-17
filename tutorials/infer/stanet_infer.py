import sys
import paddle
import os
import argparse
#加入环境
sys.path.append('./STANET_Paddle/')
import os
import argparse
import os.path as osp
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddlers.utils import logging, Timer
import paddle.inference as paddle_infer
import  cv2
import paddlers as pdrs
from paddlers import transforms as T
import paddle.nn as nn

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_file', type=str, default=3, help='save epoch')
    parser.add_argument('--model_file', type=str, default=None ,help='where model params')
    parser.add_argument('--out_dir', '-s', type=str, default=None, help='path to save result img')
    parser.add_argument('--img1', type=str, default=None, help='img1 url')
    parser.add_argument('--img2', type=str, default=None ,help='img2 url')
    parser.add_argument('--infer_dir', type=str, default=None, help='path to save result img')

    parser.add_argument('--use_gpu', type=bool, default=False, help='whether to use gpu')
    parser.add_argument('--gpu_mem', type=int, default=8000, help='gpu memory size')
    return parser
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    EXP_DIR = args.out_dir

    # 创建 config
    config = paddle_infer.Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle1 = predictor.get_input_handle(input_names[1])
    eval_transforms = T.Compose([
        T.Resize(target_size=256),
        T.Normalize(
          mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = {'image_t1': args.img1, 'image_t2': args.img2}
    image = eval_transforms(image)
    t1 = image["image"].transpose((2, 0, 1))
    t2 = image["image2"].transpose((2, 0, 1))
    t1 = paddle.to_tensor(t1).unsqueeze(0).numpy().astype("float32")
    t2 = paddle.to_tensor(t2).unsqueeze(0).numpy().astype("float32")



    # 设置输入
    # fake_input = np.random.randn(args.batch_size, 3, 256, 256).astype("float32")
    # fake_input1 = np.random.randn(args.batch_size, 3, 256, 256).astype("float32")

    input_handle.reshape(t1.shape)
    input_handle.copy_from_cpu(t1)

    input_handle1.reshape(t2.shape)
    input_handle1.copy_from_cpu(t2)

    # 运行predictor
    predictor.run()
    # 获取输出
    output_names = predictor.get_output_names()
    output_data0 = predictor.get_output_handle(output_names[0]).copy_to_cpu() # numpy.ndarray类型

    output_data0 = output_data0.astype("uint8")*255
    cv2.imwrite(EXP_DIR+"/result.png",output_data0[0])
    print('finish!')
    
  
