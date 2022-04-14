import sys
sys.path.append("./PaddleRS")

import cv2
import paddle
from paddlers import transforms as T
from paddlers.custom_models.cd.models import CDNet
import matplotlib.pyplot as plt
import argparse

transforms = T.Compose([
    T.Resize(target_size=256),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])




def predict(args):
    model = CDNet(6, 2)
    state_dict_path = args.weight
    state_dict = paddle.load(state_dict_path)
    model.set_state_dict(state_dict)
    image = {'image_t1': args.A, 'image_t2': args.B}
    image = transforms(image)
    print()
    t1 = image["image"].transpose((2, 0, 1))
    t2 = image["image2"].transpose((2, 0, 1))
    t1 = paddle.to_tensor(t1).unsqueeze(0)
    t2 = paddle.to_tensor(t2).unsqueeze(0)



    # pred = model(t1, t2)[0].squeeze(0)[1]
    # pred = pred.numpy()
    # vis = pred > 0

    pred = paddle.argmax(model(t1, t2)[-1],1)[0].numpy()
    vis = pred.astype("uint8")*255
    cv2.imwrite(args.pre,vis)
    


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--weight", type=str, required=True, \
                    help="The path of weight.")
parser.add_argument("--A", type=str, required=True, \
                    help="The path of T1 image.")
parser.add_argument("--B", type=str, required=True, \
                    help="The path of T2 image")
parser.add_argument("--pre", type=str, required=True,\
                    help="The path to save predict image.")


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args)




