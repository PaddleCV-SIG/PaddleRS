# 基于Paddle复现，并入PaddleRS套件
## 1.论文简介
STANET: [A Spatial-T emporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)，

<img src=./docs/stanetmodel.png></img>

STANet管道。已经设计了两种类型的自我注意模块。(b)特征提取器。(c)基本时空注意模块(BAM)。(d)金字塔时空注意模块(PAM)

**参考实现**：https://github.com/justchenhao/STANet

## 2.复现精度

在CDDD的测试集的测试效果如下表,达到验收指标，F1-Score=0.873


| Network | opt | epoch | batch_size | dataset | categoryF1-Score | category_iou |
| --- | --- | --- | --- | --- | --- | --- |
| STANET | AdamW  | 100 | 16 | LEVIR | **0.8753005** | 0.77825277 |


精度和loss可以用visualDL在`output\snunet\vdl_log\vdlrecords.1649682194.log`中查看。

## 3.环境依赖
通过以下命令安装对应依赖
```shell
cd STANET_Paddle/
pip install -r requirements.txt
```

## 4.数据集

下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/136610](https://aistudio.baidu.com/aistudio/datasetdetail/136610)

数据集下载解压后需要生成.txt文件用于训练。执行以下命令。

```shell
python ./data/process_cdd_data.py --data_dir=../work/Real/subset

# 切片
python /home/aistudio/PaddleRS/tools/spliter-cd.py --image_folder data/LEVIR-CD --block_size 256 --save_folder dataset
# 创建列表
python /home/aistudio/PaddleRS/tools/create_list.py --image_folder ./dataset/train --A A --B B --label label --save_txt train.txt
python /home/aistudio/PaddleRS/tools/create_list.py --image_folder ./dataset/val --A A --B B --label label --save_txt val.txt
python /home/aistudio/PaddleRS/tools/create_list.py --image_folder ./dataset/test --A A --B B --label label --save_txt test.txt



```


## 5.快速开始

### 模型训练

运行一下命令进行模型训练，在训练过程中会对模型进行评估，启用了VisualDL日志功能，运行之后在`/output/snunet/vdl_log` 文件夹下找到对应的日志文件

```shell
python tutorials/train/snunet.py --data_dir=../work/Real/subset --out_dir=./output/snunet/
```

**参数介绍**：

- data_dir:数据集路径

- out_dir:模型输出文件夹

其他超参数在snunet.py文件中已经设置好。最后一个epoch结束，模型验证日志如下：

```shell
2022-04-12 09:43:17 [INFO]      [TRAIN] Epoch=100/100, Step=25/625, loss=0.017548, lr=0.000000, time_each_step=0.73s, eta=0:7:15
2022-04-12 09:43:34 [INFO]      [TRAIN] Epoch=100/100, Step=50/625, loss=0.014375, lr=0.000000, time_each_step=0.71s, eta=0:6:46
2022-04-12 09:43:52 [INFO]      [TRAIN] Epoch=100/100, Step=75/625, loss=0.018347, lr=0.000000, time_each_step=0.71s, eta=0:6:28
2022-04-12 09:44:10 [INFO]      [TRAIN] Epoch=100/100, Step=100/625, loss=0.015912, lr=0.000000, time_each_step=0.71s, eta=0:6:11
2022-04-12 09:44:27 [INFO]      [TRAIN] Epoch=100/100, Step=125/625, loss=0.017325, lr=0.000000, time_each_step=0.71s, eta=0:5:53
2022-04-12 09:44:45 [INFO]      [TRAIN] Epoch=100/100, Step=150/625, loss=0.026069, lr=0.000000, time_each_step=0.71s, eta=0:5:35
2022-04-12 09:45:03 [INFO]      [TRAIN] Epoch=100/100, Step=175/625, loss=0.012108, lr=0.000000, time_each_step=0.71s, eta=0:5:17
2022-04-12 09:45:20 [INFO]      [TRAIN] Epoch=100/100, Step=200/625, loss=0.011837, lr=0.000000, time_each_step=0.71s, eta=0:4:59
2022-04-12 09:45:38 [INFO]      [TRAIN] Epoch=100/100, Step=225/625, loss=0.015929, lr=0.000000, time_each_step=0.71s, eta=0:4:42
2022-04-12 09:45:56 [INFO]      [TRAIN] Epoch=100/100, Step=250/625, loss=0.017316, lr=0.000000, time_each_step=0.71s, eta=0:4:24
2022-04-12 09:46:13 [INFO]      [TRAIN] Epoch=100/100, Step=275/625, loss=0.015706, lr=0.000000, time_each_step=0.7s, eta=0:4:6
2022-04-12 09:46:42 [INFO]      [TRAIN] Epoch=100/100, Step=300/625, loss=0.013702, lr=0.000000, time_each_step=1.15s, eta=0:6:14
2022-04-12 09:47:18 [INFO]      [TRAIN] Epoch=100/100, Step=325/625, loss=0.016726, lr=0.000000, time_each_step=1.43s, eta=0:7:9
2022-04-12 09:47:39 [INFO]      [TRAIN] Epoch=100/100, Step=350/625, loss=0.011909, lr=0.000000, time_each_step=0.85s, eta=0:3:53
2022-04-12 09:47:57 [INFO]      [TRAIN] Epoch=100/100, Step=375/625, loss=0.021939, lr=0.000000, time_each_step=0.7s, eta=0:2:56
2022-04-12 09:48:14 [INFO]      [TRAIN] Epoch=100/100, Step=400/625, loss=0.012447, lr=0.000000, time_each_step=0.7s, eta=0:2:38
2022-04-12 09:48:32 [INFO]      [TRAIN] Epoch=100/100, Step=425/625, loss=0.014726, lr=0.000000, time_each_step=0.71s, eta=0:2:21
2022-04-12 09:48:49 [INFO]      [TRAIN] Epoch=100/100, Step=450/625, loss=0.014930, lr=0.000000, time_each_step=0.7s, eta=0:2:3
2022-04-12 09:49:07 [INFO]      [TRAIN] Epoch=100/100, Step=475/625, loss=0.021545, lr=0.000000, time_each_step=0.7s, eta=0:1:45
2022-04-12 09:49:25 [INFO]      [TRAIN] Epoch=100/100, Step=500/625, loss=0.022362, lr=0.000000, time_each_step=0.7s, eta=0:1:27
2022-04-12 09:49:42 [INFO]      [TRAIN] Epoch=100/100, Step=525/625, loss=0.022753, lr=0.000000, time_each_step=0.7s, eta=0:1:10
2022-04-12 09:50:00 [INFO]      [TRAIN] Epoch=100/100, Step=550/625, loss=0.018402, lr=0.000000, time_each_step=0.7s, eta=0:0:52
2022-04-12 09:50:17 [INFO]      [TRAIN] Epoch=100/100, Step=575/625, loss=0.016549, lr=0.000000, time_each_step=0.7s, eta=0:0:35
2022-04-12 09:50:35 [INFO]      [TRAIN] Epoch=100/100, Step=600/625, loss=0.016990, lr=0.000000, time_each_step=0.7s, eta=0:0:17
2022-04-12 09:50:53 [INFO]      [TRAIN] Epoch=100/100, Step=625/625, loss=0.019271, lr=0.000000, time_each_step=0.7s, eta=0:0:0
2022-04-12 09:50:53 [INFO]      [TRAIN] Epoch 100 finished, loss=0.017707815 .
2022-04-12 09:50:53 [WARNING]   Segmenter only supports batch_size=1 for each gpu/cpu card during evaluation, so batch_size is forcibly set to 1.
2022-04-12 09:50:53 [INFO]      Start to evaluate(total_samples=3000, total_steps=3000)...
2022-04-12 09:52:24 [INFO]      [EVAL] Finished, Epoch=100, miou=0.951179, category_iou=[0.98762963 0.91472823], oacc=0.989079, category_acc=[0.99284724 0.96190638], kappa=0.949242, category_F1-score=[0.99377632 0.95546534] .
```
达到验收指标。


### 模型验证

除了可以再训练过程中验证模型精度，可以使用eval_snunet.py脚本进行测试，权重文件可在[百度云盘下载](https://pan.baidu.com/s/1tT6g0_B0oKQOO6AOPsJ4DA)，提取码:1tea 

```shell
python tutorials/eval/snunet_eval.py --data_dir=../work/Real/subset --weight_path=./output/snunet/best_model/model.pdparams
```
**参数介绍**：

- data_dir:数据集路径

- weight_path:模型权重所在路径

输出如下：

```shell
[04-12 12:24:42 MainThread @logger.py:242] Argv: tutorials/eval/snunet_eval.py --data_dir=../work/Real/subset --weight_path=./output/snunet/best_model/model.pdparams
[04-12 12:24:42 MainThread @utils.py:79] WRN paddlepaddle version: 2.2.2. The dynamic graph version of PARL is under development, not fully tested and supported
2022-04-12 12:24:42 [INFO]	10000 samples in file ../work/Real/subset/train.txt
2022-04-12 12:24:42 [INFO]	3000 samples in file ../work/Real/subset/test.txt
W0412 12:24:42.909528 21281 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0412 12:24:42.913775 21281 device_context.cc:465] device: 0, cuDNN Version: 7.6.
2022-04-12 12:24:45 [INFO]	Loading pretrained model from ./output/snunet/best_model/model.pdparams
2022-04-12 12:24:45 [INFO]	There are 186/186 variables loaded into SNUNet.
2022-04-12 12:24:45 [INFO]	Start to evaluate(total_samples=3000, total_steps=3000)...
OrderedDict([('miou', 0.9511789327930941), ('category_iou', array([0.98762963, 0.91472823])), ('oacc', 0.989078862508138), ('category_acc', array([0.99284724, 0.96190638])), ('kappa', 0.9492419817634077), ('category_F1-score', array([0.99377632, 0.95546534]))])
```


### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


```shell
bash ./test_tipc/prepare.sh test_tipc/configs/SNUNET/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/SNUNET/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示，也可以在`test_tipc/ouput/SNUNET`文件夹下查看：

<img src=./docs/images/TIPC_result.png></img>

## 6.代码结构与详细说明

```
SNUNet-Paddle
├── deploy               # 部署相关的文档和脚本
├── docs                 # 整个项目图片
├── output               # 输出的VDL日志
├── data                 # 数据预处理生成.txt代码
├── paddlers  
│     ├── custom_models  # 自定义网络模型代码
│     ├── datasets       # 数据加载相关代码
│     ├── models         # 套件网络模型代码
│     ├── tasks          # 相关任务代码
│     ├── tools          # 相关脚本
│     ├── transforms     # 数据处理及增强相关代码
│     └── utils          # 各种实用程序文件
├── tools                # 用于处理遥感数据的脚本
└── tutorials
      └── train          # 模型训练
      └── eval           # 模型评估和TIPC训练
      └── infer          # 模型推理


```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| SNUNET-CD |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 遥感图像变化检测 |
