import os
import os.path as osp
import argparse
from math import ceil
from PIL import Image
import numpy as np



def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename);
    return shotname, extension


def split_data_cd(image_folder, block_size, save_folder):
    img_ext = [".jpg", ".png"]
    for root, dirs, files in os.walk(image_folder):
        for dir in dirs:
            a = root[len(image_folder) + 1:]
            structure = os.path.join(save_folder, a, dir)
            os.makedirs(structure, exist_ok=True)
        for file in files:
            shotname, extension = GetFileNameAndExt(file)
            if extension in img_ext:
                file_path = osp.join(root, file)
                img_obj = Image.open(file_path)
                img_array = np.array(img_obj, dtype=np.uint8)
                rows = ceil(img_array.shape[0] / block_size)
                cols = ceil(img_array.shape[1] / block_size)
                total_number = int(rows * cols)
                for r in range(rows):
                    for c in range(cols):
                        if len(img_array.shape) > 2:
                            title = Image.fromarray(
                                img_array[r * block_size:(r + 1) * block_size,c * block_size:(c + 1) * block_size, :])
                            save_path = osp.join(root.replace(image_folder, save_folder),
                                                 (shotname + "_" + str(r) + "_" + str(c) + ".png"))
                        else:
                            title = Image.fromarray(
                                img_array[r * block_size:(r + 1) * block_size,c * block_size:(c + 1) * block_size])
                            save_path = osp.join(root.replace(image_folder, save_folder),
                                                 (shotname + "_" + str(r) + "_" + str(c) + ".png"))
                        title.save(save_path, "PNG")
                        print("-- {:d}/{:d} --".format(int(r * cols + c + 1), total_number))



parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--image_folder", type=str, required=True, \
                    help="The path of big image data.")
parser.add_argument("--block_size", type=int, default=512, \
                    help="The size of image block, `512` is the default.")
parser.add_argument("--save_folder", type=str, default="output", \
                    help="The folder path to save the results, `output` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    split_data_cd(args.image_folder, args.block_size, args.save_folder)
    # split_data_cd("/home/dl/下载/levir-cd", 256, "./dataset")
