![静默活体检测](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  
该项目为[小视科技](https://www.minivision.cn/)的静默活体检测项目,您可以扫描下方的二维码获取APK(提取码为**phwd**),在安卓端安装APK,体验静默活体的检测效果.  
![静默活体APK](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/静默活体APK.png)  
我们的[AI开放平台](https://ai.minivision.cn/#/coreability/livedetection)也支持在线体验。  
# 静默活体检测 (Silent-Face-Anti-Spoofing)   
## 简介
  活体检测技术旨在人脸识别之前抵御假脸的恶意攻击，攻击方式括纸质照片，电脑屏幕，面具，3D打印模型等。可分为配合式活体检测和非配合式活体检测（静默活体检测）。配合式活体检测需要用户根据提示完成指定的动作或者表情变化然后通过算法判断当前用户是否为真人，静默活体则只需要采集一张用户照片即可输出活体结果。  
因傅里叶频谱图一定程度上能够反应真假脸在频域的差异,我们提出了一种深度学习结合图像傅里叶变换的静默活体检测方法, 模型架构由分类主分支和傅里叶频谱图辅助监督分支构成，整体架构如下图所示：  
![整体架构图](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/framework.png)  

因为移动端的算力有限,为了最大程度的降低模型开销,提升用户的体验.采用剪枝的MobileFaceNet作为backbone,在精度稍有损失的情况下,明显提升模型的性能(降低计算量与参数量).  

|Model|FLOPs|Params|
| :------:|:-----:|:-----:| 
|MobileFaceNet|0.224G|0.991M|
|MobileFaceNetPv1|0.081G|0.414M|
|MobileFaceNetPv4|0.081G|0.435M|

## APK
 
### 关键指标  
| Model|FLOPs|Speed| FPR | TPR |备注 |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK模型 |84M| 20ms | 1e-5|97.8%| 开源|
| 高精度模型 |162M| 40ms| 1e-5 |99.7%| 未开源 |

### Demo  
![demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif)

### 测试方法  
- 显示信息:速度(ms), 置信度(0~1)以及活体检测结果(真脸or假脸)
- 点击右上角图标可设置阈值,如下图所示,如果置信度大于阈值,为真脸,否则为假脸  
![设置阈值](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/设置阈值.png)

### 测试须知 
- 所有测试图片必须通过摄像头采集得到，否则不符合正常场景使用规范，算法效果也无法保证。
- 因为RGB静默活体对摄像头型号和使用场景鲁棒性受限，所以实际使用体验会有一定差异。
- 测试时，应保证有完整的人脸出现在视图中，并且人脸旋转角与竖直方向小于30度（符合正常刷脸场景），否则影响体验。　　

**已测试手机型号**

|手机型号|荣耀V30 pro |华为mate30 |华为nova5 pro |一加7 pro |小米8 |
| :------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|速度/ms|19|23|25|25|24|

## 工程
### 安装依赖库  
`pip install -r requirements.txt`
## Clone
```
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing  
cd Silent-Face-Anti-Spoofing
```  
## 训练  
### 数据预处理
1.将训练集分为3类,分别为纸质假脸(0类),真脸(1类)和电子屏幕假脸(2类),相同类别的图片放入一个文件夹;  
2.因采用多尺度模型融合的方法,分别用原图和不同的patch训练模型,所以将数据分为原图和基于原图的patch;  
- 原图(org_1_height**x**width),直接将原图resize到固定尺寸(width, height),如图1所示;  
- 基于原图的patch(scale_height**x**width),采用人脸检测器人脸,获取人脸框,按照一定比例(scale)对人脸框进行扩边，为了保证模型的输入尺寸的一致性，将人脸框区域resize到固定尺寸(width, height),图2-4分别显示了scale为1,2.7和4的patch样例;  
![patch demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)  

3.采用傅里叶频谱图作为辅助监督,训练集图片均需生成对应的傅里叶频谱图.
**数据集的目录结构如下所示**
```
├── datasets
    ├── RGB_Images
        ├── org_1_80x60
            ├── 0
            ├── 1
            └── 2
        ├── 1_80x80
        └── ...
    └── Fourier_Images
        ├── 0
        ├── 1
        └── 2
```
### 训练
```
python train.py --device_ids 0  --patch_info your_patch
```  
### 测试
 ./resources/anti_spoof_models 活体检测的融合模型  
 ./resources/detection_model 检测器模型  
 ./images/sample 测试图片  
 **注意:测试图片需从宽高比为3:4的视频流中截取**   
 ```
 python test.py --image_name your_image_name
```  
## Q&A
**Q:**  数据集开源吗?  
A: 不开源  
**Q:** 为什么测试图片需从宽高比为3:4的视频流中截取?  
A: 因为活体的应用场景是实时视频流,该活体模型适配480x640的实时视频流  
 
## 参考  
- 检测器 [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
