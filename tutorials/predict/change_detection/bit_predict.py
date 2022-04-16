# 变化检测模型SNUNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('../PaddleRS')
import paddle
import cv2
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_path',
        type=str,
        default=r"../work/output/BIT/best_model/model.pdparams",
        help='weight path')
    parser.add_argument("--A", type=str, required=True, \
                    help="The path of T1 image.")
    parser.add_argument("--B", type=str, required=True, \
                    help="The path of T2 image")
    parser.add_argument("--pre", type=str, required=True,\
                    help="The path to save predict image.")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    state_dict_path = args.weight_path

    transforms = T.Compose([
        T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 验证阶段与训练阶段的数据归一化方式必须相同
    ])

    # 使用默认参数构建SNUNet模型
    model = pdrs.tasks.BIT()

    model.net_initialize(pretrain_weights=state_dict_path)
    print('ok')
    model.net.eval()
    image = {'image_t1': args.A, 'image_t2': args.B}
    image = transforms(image)
    t1 = image["image"].transpose((2, 0, 1))
    t2 = image["image2"].transpose((2, 0, 1))
    t1 = paddle.to_tensor(t1).unsqueeze(0)
    t2 = paddle.to_tensor(t2).unsqueeze(0)
    vis = paddle.argmax(model.net(t1, t2)[-1], 1)[0].numpy()
    vis = vis.astype("uint8") * 255
    cv2.imwrite(args.pre, vis)
    print('finish!')
