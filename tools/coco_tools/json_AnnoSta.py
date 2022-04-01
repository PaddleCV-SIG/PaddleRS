# -*- coding: utf-8 -*- 
# @File             : json_annotations_sta.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/8/1 10:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/8/1 10:25
'''
@File Description:
# json文件annotations信息，生成统计结果csv，对象框shape、对象看shape比例、对象框起始位置、对象结束位置、对象结束位置、对象类别、单个图像对象数量的分布
!python /home/aistudio/work/json_annotation_sta.py \
    --json_path=/home/aistudio/data/data102587/instances_val2017.json \
    --csv_path=/home/aistudio/data/data102587/instances_val2017_annotations.csv \
    --png_shape_path=/home/aistudio/data/data102587/instances_val2017_annotations_shape.png \
    --png_shapeRate_path=/home/aistudio/data/data102587/instances_val2017_annotations_shapeRate.png \
    --png_pos_path=/home/aistudio/data/data102587/instances_val2017_annotations_pos.png \
    --png_posEnd_path=/home/aistudio/data/data102587/instances_val2017_annotations_posEnd.png \
    --png_cat_path=/home/aistudio/data/data102587/instances_val2017_annotations_cat.png \
    --png_objNum_path=/home/aistudio/data/data102587/instances_val2017_annotations_objNum.png \
    --get_relative=True
'''

import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


shp_rate_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1,
                 2.2, 2.4, 2.6, 3, 3.5, 4, 5]


def js_anno_sta(js_path, csv_path, png_shape_path, png_shapeRate_path, png_pos_path, png_posEnd_path, png_cat_path,
                png_objNum_path, get_relative):
    print('json read...\n')
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)

    df_img = pd.DataFrame(data['images'])
    sns.jointplot('height', 'width', data=df_img, kind='hex')
    plt.close()
    df_img = df_img.rename(columns={"id": "image_id", "height": "image_height", "width": "image_width"})

    df_anno = pd.DataFrame(data['annotations'])
    df_anno[['pox_x', 'pox_y', 'width', 'height']] = pd.DataFrame(df_anno['bbox'].values.tolist())
    df_anno['width'] = df_anno['width'].astype(int)
    df_anno['height'] = df_anno['height'].astype(int)

    df_merge = pd.merge(df_img, df_anno, on="image_id")

    if png_shape_path is not None:
        sns.jointplot('height', 'width', data=df_merge, kind='hex')
        plt.savefig(png_shape_path)
        plt.close()
        print('png save to', png_shape_path)
        if get_relative:
            png_shapeR_path = png_shape_path.replace('.png', '_Relative.png')
            df_merge['heightR'] = df_merge['height'] / df_merge['image_height']
            df_merge['widthR'] = df_merge['width'] / df_merge['image_width']
            sns.jointplot('heightR', 'widthR', data=df_merge, kind='hex')
            plt.savefig(png_shapeR_path)
            plt.close()
            print('png save to', png_shapeR_path)
    if png_shapeRate_path is not None:
        plt.figure(figsize=(12, 8))
        df_merge['shape_rate'] = (df_merge['width'] / df_merge['height']).round(1)
        df_merge['shape_rate'].value_counts(sort=False, bins=shp_rate_bins).plot(kind='bar', title='images shape rate')
        plt.xticks(rotation=20)
        plt.savefig(png_shapeRate_path)
        plt.close()
        print('png save to', png_shapeRate_path)

    if png_pos_path is not None:
        sns.jointplot('pox_y', 'pox_x', data=df_merge, kind='hex')
        plt.savefig(png_pos_path)
        plt.close()
        print('png save to', png_pos_path)
        if get_relative:
            png_posR_path = png_pos_path.replace('.png', '_Relative.png')
            df_merge['pox_yR'] = df_merge['pox_y'] / df_merge['image_height']
            df_merge['pox_xR'] = df_merge['pox_x'] / df_merge['image_width']
            sns.jointplot('pox_yR', 'pox_xR', data=df_merge, kind='hex')
            plt.savefig(png_posR_path)
            plt.close()
            print('png save to', png_posR_path)
    if png_posEnd_path is not None:
        df_merge['pox_y_end'] = df_merge['pox_y'] + df_merge['height']
        df_merge['pox_x_end'] = df_merge['pox_x'] + df_merge['width']
        sns.jointplot('pox_y_end', 'pox_x_end', data=df_merge, kind='hex')
        plt.savefig(png_posEnd_path)
        plt.close()
        print('png save to', png_posEnd_path)
        if get_relative:
            png_posEndR_path = png_posEnd_path.replace('.png', '_Relative.png')
            df_merge['pox_y_endR'] = df_merge['pox_y_end'] / df_merge['image_height']
            df_merge['pox_x_endR'] = df_merge['pox_x_end'] / df_merge['image_width']
            sns.jointplot('pox_y_endR', 'pox_x_endR', data=df_merge, kind='hex')
            plt.savefig(png_posEndR_path)
            plt.close()
            print('png save to', png_posEndR_path)

    if png_cat_path is not None:
        plt.figure(figsize=(12, 8))
        df_merge['category_id'].value_counts().sort_index().plot(kind='bar', title='obj category')
        plt.savefig(png_cat_path)
        plt.close()
        print('png save to', png_cat_path)

    if png_objNum_path is not None:
        plt.figure(figsize=(12, 8))
        df_merge['image_id'].value_counts().value_counts().sort_index().plot(kind='bar', title='obj number per image')
        # df_merge['image_id'].value_counts().value_counts(bins=np.linspace(1,31,16)).sort_index().plot(kind='bar', title='obj number per image')
        plt.xticks(rotation=20)
        plt.savefig(png_objNum_path)
        plt.close()
        print('png save to', png_objNum_path)

    if csv_path is not None:
        df_merge.to_csv(csv_path)
        print('csv save to', csv_path)


def get_args():
    parser = argparse.ArgumentParser(description='Json Images Infomation Statistic')

    # parameters
    parser.add_argument('--json_path', type=str,
                        help='json path to statistic images information')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='csv path to save statistic images information, default None, do not save')

    parser.add_argument('--png_shape_path', type=str, default=None,
                        help='png path to save statistic images shape information, default None, do not save')
    parser.add_argument('--png_shapeRate_path', type=str, default=None,
                        help='png path to save statistic images shape rate information, default None, do not save')

    parser.add_argument('--png_pos_path', type=str, default=None,
                        help='png path to save statistic pos information, default None, do not save')
    parser.add_argument('--png_posEnd_path', type=str, default=None,
                        help='png path to save statistic end pos information, default None, do not save')

    parser.add_argument('--png_cat_path', type=str, default=None,
                        help='png path to save statistic category information, default None, do not save')
    parser.add_argument('--png_objNum_path', type=str, default=None,
                        help='png path to save statistic images object number information, default None, do not save')

    parser.add_argument('--get_relative', type=bool, default=True,
                        help='if True, get relative result')

    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100, '-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args


if __name__ == '__main__':
    args = get_args()
    js_anno_sta(args.json_path, args.csv_path, args.png_shape_path, args.png_shapeRate_path,
                args.png_pos_path, args.png_posEnd_path, args.png_cat_path, args.png_objNum_path, args.get_relative)


