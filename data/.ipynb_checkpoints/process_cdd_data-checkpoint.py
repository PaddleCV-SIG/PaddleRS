import random
import os.path as osp
from glob import glob
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  type=str, default=None, help='data directory path')
    return parser

def write_rel_paths(phase, names, out_dir, prefix=''):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            f.write(
                ' '.join([
                    osp.join(prefix, 'A', name),
                    osp.join(prefix, 'B', name),
                    osp.join(prefix, 'OUT', name)
                ])
            )
            f.write('\n')


if __name__ == "__main__":
   # 数据集路径
    parser = get_parser()
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    # 随机数生成器种子
    RNG_SEED = 114514
    random.seed(RNG_SEED)

    write_rel_paths(
        'train', 
        map(osp.basename, glob(osp.join(DATA_DIR, 'train', 'OUT', '*.jpg'))), 
        DATA_DIR, 
        prefix='train'
    )

    write_rel_paths(
        'val', 
        map(osp.basename, glob(osp.join(DATA_DIR, 'val', 'OUT', '*.jpg'))), 
        DATA_DIR, 
        prefix='val'
    )

    write_rel_paths(
        'test', 
        map(osp.basename, glob(osp.join(DATA_DIR, 'test', 'OUT', '*.jpg'))), 
        DATA_DIR,
        prefix='test'
    )

    print("数据集划分已完成。")