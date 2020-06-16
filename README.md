![image](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/image_T1.jpg)  
该项目为[小视科技](https://www.minivision.cn/)的静默活体检测项目,您可以在安卓端安装下方的APK,体验静默活体的检测效果,我们的[AI开放平台](https://ai.minivision.cn/#/coreability/livedetection)也支持在线体验。
# 静默活体检测 (Silent-Face-Anti-Spoofing)   
## 简介
活体检测技术旨在解决人脸识别过程中的假脸攻击（主要包括纸质照片，电脑屏幕，面具，3D打印假脸等）问题，可分为配合式活体检测和非配合式活体检测（静默活体检测）。配合式活体检测需要用户根据提示完成指定的动作或者表情变化然后通过算法判断当前用户是否为真人，静默活体则只需要采集一张用户照片即可输出活体结果。  
我们静默活体的算法采用深度学习结合图像傅里叶变换的方法, 模型架构由分类主分支和辅助分支构成，整体架构如下图所示：
![整体架构图]()

## APK
 
### 关键指标  
| Model|Speed| FPR | TPR |
| :------:| :-----: | :----: | :----: |
| 高精度模型 | 40ms| 1e-5 |99.7%|
|   APK模型 | 20ms | 1e-5|97.8%|

### 测试须知 
- 所有测试图片必须通过摄像头采集得到，否则不符合正常场景使用规范，算法效果也无法保证。
- 因为RGB静默活体对摄像头型号和使用场景鲁棒性受限，所以实际使用体验会有一定差异。
- 测试时，应保证有完整的人脸出现在视图中，并且人脸旋转角与竖直方向小于30度（符合正常刷脸场景），否则影响体验。


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
1.将训练集分为3类,分别为纸质假脸,真脸和电子屏幕假脸,相同类别的图片放入一个文件夹。
2.模型采用多尺度输入,输入类型为原图和patch.
3.对图片进行切patch,采用人脸检测器图像中的人脸,获取人脸框,按照一定比例(scale)对人脸框进行扩边，为了保证模型的输入尺寸的一致性，将人脸框区域resize到固定尺寸(width, height)。下图中展示了部分patch的区域，参数信息2.7_0_0_80x80，分别表示scale，shift_ratio_x， shift_ratio_y，width,  height.
![不同的patch展示图]()
## 训练
`python train.py --device_ids 0 --train_set your_trainset_name --num_classes 3  --patch_info your_patch`
device_ids:选择GPU，可以为多个，比如0123
train_set:训练集的名称
patch_info:选择用于训练的patch
##测试
`python test.py --image_name “ ”`
 
## 技术交流  
