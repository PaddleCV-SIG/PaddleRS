# 变化检测模型SNUNet训练示例脚本
# 执行此脚本前，请确认已正确安装PaddleRS库
import sys
sys.path.append('../PaddleRS')
import paddle
import os
import argparse
import paddlers as pdrs
from paddlers import transforms as T


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', '-m', type=str, default=None, help='model directory path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--weight_path',
        type=str,
        default=r"./output/BIT/best_model/model.pdparams",
        help='weight path')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST_PATH = os.path.join(DATA_DIR, 'train.txt')
    EVAL_FILE_LIST_PATH = os.path.join(DATA_DIR, 'test.txt')
    #测试阶段 batch size
    EVAL_BATCH_SIZE = args.batch_size
    state_dict_path = args.weight_path

    eval_transforms = T.Compose([
        T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 验证阶段与训练阶段的数据归一化方式必须相同
    ])

    eval_dataset = pdrs.datasets.CDDataset(
        data_dir=DATA_DIR,
        file_list=EVAL_FILE_LIST_PATH,
        label_list=None,
        transforms=eval_transforms,
        num_workers=0,
        shuffle=False,
        with_seg_labels=False,
        binarize_labels=True)

    # 使用默认参数构建SNUNet模型
    model = pdrs.tasks.BIT()

    model.net_initialize(pretrain_weights=state_dict_path)
    eval_metrics = model.evaluate(eval_dataset)
    print(str(eval_metrics))
