# 基于Paddle复现
## 1.论文简介
STANET: [A Spatial-T emporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection](https://www.mdpi.com/2072-4292/12/10/1662)，

<img src=./docs/stanetmodel.png></img>

本次复现的模型为时空注意神经网络(STANet)。
STANet设计了两种类型的自我注意模块。基本时空注意模块(BAM)。金字塔时空注意模块(PAM)

**参考实现**：https://github.com/justchenhao/STANet

## 2.复现精度

在LEVIR的测试集的测试效果如下表,达到验收指标，F1-Score=0.873


| Network | opt | epoch | batch_size | dataset | categoryF1-Score | category_iou |
| --- | --- | --- | --- | --- | --- | --- |
| STANET | AdamW  | 100 | 8 | LEVIR | **0.8753005** | 0.77825277 |


精度和loss可以用visualDL在`output\stanet\vdl_log\vdlrecords.1649956922.log`中查看。

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
#切片
python ./STANET_Paddle/tools/spliter-cd.py  --image_folder  data/LEVIR-CD --block_size 256 --save_folder dataset
```
**参数介绍**：

- image_folder:数据集路径
- block_size:切片大小
- save_folder:保存路径
 ```shell
# 创建列表
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/train --A A --B B --label label --save_txt train.txt
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/val --A A --B B --label label --save_txt val.txt
python ./STANET_Paddle/tools/create_list.py --image_folder ./dataset/test --A A --B B --label label --save_txt test.txt
```
**参数介绍**：

- image_folder:切片后数据集路径
- -A  -B  -label :A时相、B时相、label的子路径名
- save_txt:保存名

## 5.快速开始

### 模型训练

运行一下命令进行模型训练，在训练过程中会对模型进行评估，启用了VisualDL日志功能，运行之后在`/output/stanet/vdl_log` 文件夹下找到对应的日志文件

```shell
!python ./STANET_Paddle/tutorials/train/stanet_train.py --data_dir=./dataset/   --out_dir=./output/stanet/   --batch_size=8 
```

**参数介绍**：

- data_dir:数据集路径

- out_dir:模型输出文件夹
- batch_size：batch大小

其他超参数已经设置好。最后一个epoch结束，模型验证日志如下：
```shell
2022-04-15 07:27:06 [INFO]	[TRAIN] Epoch=100/100, Step=730/890, loss=0.039839, lr=0.000000, time_each_step=0.23s, eta=0:0:36
2022-04-15 07:27:11 [INFO]	[TRAIN] Epoch=100/100, Step=750/890, loss=0.012171, lr=0.000000, time_each_step=0.23s, eta=0:0:31
2022-04-15 07:27:15 [INFO]	[TRAIN] Epoch=100/100, Step=770/890, loss=0.003880, lr=0.000000, time_each_step=0.23s, eta=0:0:27
2022-04-15 07:27:20 [INFO]	[TRAIN] Epoch=100/100, Step=790/890, loss=0.008041, lr=0.000000, time_each_step=0.23s, eta=0:0:22
2022-04-15 07:27:24 [INFO]	[TRAIN] Epoch=100/100, Step=810/890, loss=0.011668, lr=0.000000, time_each_step=0.22s, eta=0:0:17
2022-04-15 07:27:29 [INFO]	[TRAIN] Epoch=100/100, Step=830/890, loss=0.015986, lr=0.000000, time_each_step=0.23s, eta=0:0:13
2022-04-15 07:27:33 [INFO]	[TRAIN] Epoch=100/100, Step=850/890, loss=0.017611, lr=0.000000, time_each_step=0.23s, eta=0:0:9
2022-04-15 07:27:38 [INFO]	[TRAIN] Epoch=100/100, Step=870/890, loss=0.005013, lr=0.000000, time_each_step=0.22s, eta=0:0:4
2022-04-15 07:27:42 [INFO]	[TRAIN] Epoch=100/100, Step=890/890, loss=0.011437, lr=0.000000, time_each_step=0.22s, eta=0:0:0
2022-04-15 07:27:43 [INFO]	[TRAIN] Epoch 100 finished, loss=0.01370322 .
2022-04-15 07:27:43 [WARNING]	Segmenter only supports batch_size=1 for each gpu/cpu card during evaluation, so batch_size is forcibly set to 1.
2022-04-15 07:27:43 [INFO]	Start to evaluate(total_samples=1024, total_steps=1024)...
2022-04-15 07:28:37 [INFO]	[EVAL] Finished, Epoch=100, miou=0.874413, category_iou=[0.98915931 0.75966758], oacc=0.989518, category_acc=[0.99084709 0.95264434], kappa=0.858022, category_F1-score=[0.99455012 0.86342169] .
2022-04-15 07:28:37 [INFO]	Current evaluated best model on eval_dataset is epoch_78, miou=0.8840882464473624
2022-04-15 07:28:38 [INFO]	Model saved in output/stanet/epoch_100.
```
最好的结果在第78个epoch test时达到验收指标。


### 模型验证

除了可以再训练过程中验证模型精度，可以使用eval_stanet.py脚本进行测试，权重文件可在[百度云盘下载](https://pan.baidu.com/s/1f7DRYNOxfcnxxVCv11HzdQ)，提取码:8c0c 

```shell
!python ./STANET_Paddle/tutorials/eval/stanet_eval.py --data_dir=./dataset/   --state_dict_path=./output/home/aistudio/output/stanet/best_model/model.pdparams
```
**参数介绍**：

- data_dir:数据集路径

- weight_path:模型权重所在路径

输出如下：

```shell
2022-04-15 08:50:50 [INFO]	1024 samples in file val.txt
2022-04-15 08:50:51 [INFO]	Loading pretrained model from output/stanet/best_model/model.pdparams
2022-04-15 08:50:51 [INFO]	There are 186/186 variables loaded into STANet.
2022-04-15 08:50:51 [INFO]	Start to evaluate(total_samples=1024, total_steps=1024)...
OrderedDict([('miou', 0.8840882464473624), ('category_iou', array([0.98992372, 0.77825277])), ('oacc', 0.990267887711525), ('category_acc', array([0.99189824, 0.94670973])), ('kappa', 0.8702668237858971), ('category_F1-score', array([0.99493635, 0.8753005 ]))])
```



### 导出

可以将模型导出，动态图转静态图，使模型预测更快，可以使用stanet_export.py脚本进行测试

在这里因为动静态模型转化的问题，修改了stanet的模型代码使其可以转出静态模型。

调试过程中参考这份文档   [报错调试](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/debugging_cn.html)

```shell
!python ./STANET_Paddle/deploy/export/stanet_export.py    --state_dict_path=./output/home/aistudio/output/stanet/best_model/model.pdparams    --save_dir=./inference_model/  --fixed_input_shape=[1,3,256,256]
```
**参数介绍**：
- fixed_input_shape:预测图的形状
-save_dir  静态图导出路径
- state_dict_path:模型权重所在路径



### 使用静态图推理

可以使用stanet_infer.py脚本进行测试

```shell
!python ./STANET_Paddle/tutorials/infer/stanet_infer.py   --infer_dir=./inference_model   --img_dir=./STANET_Paddle/test_tipc/data/mini_levir_dataset --output_dir=./STANET_Paddle/test_tipc/result/predict_output
```
**参数介绍**：
- infer_dir:模型文件路径
- img_dir：用于推理的图片路径
- output_dir：预测结果输出路径

### 使用动态图预测

```shell
!python ./STANET_Paddle/tutorials/predict/stanet_predict.py --img1=./STANET_Paddle/test_tipc/data/mini_levir_dataset/test/A/test_1_0_3.png --img2=./STANET_Paddle/test_tipc/data/mini_levir_dataset/test/B/test_1_0_3.png   --state_dict_path=./output/home/aistudio/output/stanet/best_model/model.pdparams   --out_dir=./
```
**参数介绍**：
- img1:A时相影像路径
- img2：B时相影像路径
- state_dict_path：模型文件路径
- out_dir：预测结果输出路径
- 预测结果与真实值对比

| 前时相 | 后时相 | 预测结果 | 真值 |
|---|---|---| --- |
|![](./docs/a.png)|![](./docs/b.png) |![](./docs/result.png)| ![](./docs/label.png)|



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
bash  ./STANET_Paddle/test_tipc/prepare.sh  ./STANET_Paddle/test_tipc/configs/stanet/train_infer_python.txt 'lite_train_lite_infer'

bash  ./STANET_Paddle/test_tipc/test_train_inference_python.sh ./STANET_Paddle/test_tipc/configs/stanet/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示

<img src=./docs/TIFC_pre.png></img>

<img src=./docs/TIFC_next.png></img>
<img src=./docs/TIFC_result.png></img>

## 6.代码结构与详细说明

```
StaNet-Paddle
├── deploy               # 部署相关的文档和脚本
├── docs                 # 整个项目图片
├── output               # 输出的VDL日志
├── test_tipc            # tipc程序相关
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
      └── predict        # 动态模型预测

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| STANET |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 遥感图像变化检测 |

## 8.说明
感谢百度提供的算力，以及举办的本场比赛，让我增强对paddle的熟练度，加深对变化检测模型的理解！

## 9.参考
部分功能参考 [PaddleRS 手把手教你PaddleRS实现变化检测](https://aistudio.baidu.com/aistudio/projectdetail/3737991?channelType=0&channel=0)

部分功能参考 [基于Paddle复现SNUNet-CD](https://github.com/kongdebug/SNUNet-Paddle)，


