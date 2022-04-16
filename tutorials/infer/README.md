# Python部署

PaddleRS已经集成了基于Python的高性能预测（prediction）接口。在安装PaddleRS后，可参照如下代码示例执行预测。

## 部署模型导出

在服务端部署模型时需要首先将训练过程中保存的模型导出为部署格式，具体的导出步骤请参考文档[部署模型导出](/deploy/export/README.md)。这是PaddleRs套件自带的，通过该命令能导出模型，BIT模型由于用到空间注意力，所以需要固定尺寸。
```shell
python deploy/export/export_model.py --model_dir=../work/output/BIT/best_model/ --save_dir=./inference_model/ --fixed_input_shape=[1,3,256,256]
```

## 预测脚本说明

* **基本使用**

本文件夹中的change_detection文件夹有预测的脚本，为`bit_infer.py`，是本项目的预测脚本，可以通过以下命令使用，运行该脚本可以得到预测的速度以及导出模型后的预测结果
```shell
python tutorials/infer/change_detection/bit_infer.py --infer_dir=./inference_model/ --img_dir=./test_tipc/data/mini_dataset --output_dir=./test_tipc/result/predict_output
```
**参数说明**
- img_dir:数据所在文件夹
- infer_dir:导出的模型所在文件夹
- output_dir:保存预测的label文件夹，默认为当前文件夹下的`output`文件夹
- 此外，还有`warmup_iters`与`repeats`以下两个参数

**关于`warmup_iters`与`repeats`参数设置的目的**

加载模型后，对前几张图片的预测速度会较慢，这是因为程序刚启动时需要进行内存、显存初始化等步骤。通常，在处理20-30张图片后，模型的预测速度能够达到稳定值。基于这一观察，**如果需要评估模型的预测速度，可通过指定预热轮数`warmup_iters`对模型进行预热**。此外，**为获得更加精准的预测速度估计值，可指定重复`repeats`次预测后计算平均耗时**。
