[中文版](README.md)|**English Version**  
![Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  
# Silent-Face-Anti-Spoofing 

This project is Silent-Face-Anti-Spoofing belongs to [minivision technology](https://www.minivision.cn/). You can scan the QR code below to get APK and install it on Android side to experience the effect of real time living detection(silent face anti-spoofing detection).   
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/静默活体APK.jpeg" width="200" height="200" align=center />  

## Introduction

In this project, we open source the silent face anti-spoofing model with training architecture, data preprocessing method, model training & test script and open source APK for real time testing.  

The main purpose of silent face anti-spoofing detection technology is to judge whether the face in front of the machine is real or fake. The face presented by other media can be defined as false face, including printed paper photos, display screen of electronic products, silicone mask, 3D human image, etc. At present, the mainstream solutions includes cooperative living detection and non cooperative living detection (silent living detection). Cooperative living detection requires the user to complete the specified action according to the prompt, and then carry out the live verification, while the silent live detection directly performs the live verification.  

Since the Fourier spectrum can reflect the difference of true and false faces in frequency domain to a certain extent, we adopt a silent living detection method based on the auxiliary supervision of Fourier spectrum. The model architecture consists of the main classification branch and the auxiliary supervision branch of Fourier spectrum. The overall architecture is shown in the following figure:

![overall architecture](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg)  

By using our self-developed model pruning method, the FLOPs of MobileFaceNet is reduced from 0.224G to 0.081G, and the performance of the model is significantly improved (the amount of calculation and parameters is reduced) with little loss of precision.


|Model|FLOPs|Params|
| :------:|:-----:|:-----:| 
|MobileFaceNet|0.224G|0.991M|
|MiniFASNetV1|0.081G|0.414M|
|MiniFASNetV2|0.081G|0.435M|

## APK
### APK source code  
Open source for Android platform deployment code: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK  

### Demo
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif" width="300" height="400"/>  
 
### Performance  
| Model|FLOPs|Speed| FPR | TPR |comments |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK |84M| 20ms | 1e-5|97.8%| Open Source|
| High precision model |162M| 40ms| 1e-5 |99.7%| Private |

### Test Method 

- Display information: speed(ms), confidence(0 ~ 1) and in living detection test results (true face or false face).
- Click the icon in the upper right corner to set the threshold value. If the confidence level is greater than the threshold value, it is a true face, otherwise it is a fake face.

### Before test you must know

- All the test images must be collected by camera, otherwise it does not conform to the normal scene usage specification, and the algorithm effect cannot be guaranteed.
- Because the robustness of RGB silent living detection depending on camera model and scene, the actual use experience could be different.
- During the test, it should be ensured that a complete face appears in the view, and the rotation angle and vertical direction of the face are less than 30 degrees (in line with the normal face recognition scene), otherwise, the experience will be affected.　

**Tested mobile phone processor**

|type|Kirin990 5G|Kirin990 |Qualcomm845 |Kirin810 |RK3288 |
| :------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Speed/ms|19|23|24|25|90|

## Repo
### Install dependency Library  
```
pip install -r requirements.txt
```
### Clone
```
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing  
cd Silent-Face-Anti-Spoofing
```  
### Data Preprocessing
1.The training set is divided into three categories, and the pictures of the same category are put into a folder;  
2.Due to the multi-scale model fusion method, the original image and different patch are used to train the model, so the data is divided into the original map and the patch based on the Original picture;  
- Original picture(org_1_height**x**width),resize the original image to a fixed size (width, height), as shown in Figure 1;  
- Patch based on original(scale_height**x**width),The face detector is used to obtain the face frame, and the edge of the face frame is expanded according to a certain scale. In order to ensure the consistency of the input size of the model, the area of the face frame is resized to a fixed size (width, height). Fig. 2-4 shows the patch examples with scales of 1, 2.7 and 4;  
![patch demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)  

3.The Fourier spectrum is used as the auxiliary supervision, and the corresponding Fourier spectrum is generated online from the training set images.  
**The directory structure of the dataset is shown below**
```
├── datasets
    └── RGB_Images
        ├── org_1_80x60
            ├── 0
		├── aaa.png
		├── bbb.png
		└── ...
            ├── 1
		├── ddd.png
		├── eee.png
		└── ...
            └── 2
		├── ggg.png
		├── hhh.png
		└── ...
        ├── 1_80x80
        └── ...
```  
### Train
```
python train.py --device_ids 0  --patch_info your_patch
```  
### Test
 ./resources/anti_spoof_models Fusion model of in living detection  
 ./resources/detection_model Detector  
 ./images/sample Test Images  
 ```
 python test.py --image_name your_image_name
 ```    
## Reference 
- Detector [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)  

For this project, in order to facilitate the technical exchange of developers, we created QQ group: 1121178835, welcome to join.  

In addition to the open-source silent living detection algorithm, Minivision technology also has a number of self-developed algorithms and SDK related to face recognition and human body recognition. Interested individual developers or enterprise developers can visit our website: [Mini-AI Open Platform](https://ai.minivision.cn/)
Welcome to contact us.
