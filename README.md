![image](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/imgs/anti-face-spoofing.gif)  
# 静默活体检测 (Silent-Face-Anti-Spoofing)  
   活体检测技术旨在解决人脸识别过程中的假脸攻击（主要包括纸质照片，电脑屏幕，面具，3D打印假脸等）问题，可分为配合式活体检测和非配合式活体检测（静默活体检测）。配合式活体检测需要用户根据提示完成指定的动作或者表情变化然后通过算法判断当前用户是否为真人，静默活体则只需要采集一张用户照片即可输出活体结果。
   我们静默活体的算法采用深度学习结合图像傅里叶变换的方法, 模型架构由分类主分支和辅助分支构成，整体架构如下图所示：
# 数据预处理
   采用公司自研的人脸检测器获取图像中的人脸框坐标，按照一定比例(scale)对人脸框进行扩边，选择性地进行水平（shift_ratio_x）和垂直（shift_ratio_y）偏移。为了保证模型的输入尺寸的一致性，将人脸框区域resize到固定尺寸(width,height)。下图中展示了部分patch的区域，参数信息2.7_0_0_80x80，分别表示scale，shift_ratio_x， shift_ratio_y，width, height
# 模型训练
   将相同类别的图片放入同一文件夹，python train.py --device_ids 0 --train_set ‘your trainset’ --num_classes ‘your classes’  --patch_info ‘select patch’
# 模型测试
   准备图片放入./images文件夹，提前用检测器检测出人脸框，在test.py中IMAGE_BBOX设置图片名以及人脸框，python test.py --image_name “ ”
## 安装  
## Demo  
## 关键指标  
## 测试须知  
# 技术交流  
