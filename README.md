# Towards-Realtime-MOT-Cpp
A C++ codebase implementation of [Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT).

## Introduction
This repo is the a c++ codebase of the Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. We hope this repo will help researches/engineers to develop more practical MOT systems.

## Requirements
* Sys-Windows10 (Windows7 should also work)
* GPU-Nvidia (GTX-1080/RTX-2080/RTX-2080Ti)
* IDE-VS2017/VS2019
* cuda == 10.1, cudnn == 7.6
* LibTorch-1.4.0 [[Baidu]](https://pan.baidu.com/s/1aSKPJf9DmXGlEujfm93BEA) (PWD: gr3t)
* OpenCV == 4.2.0
* eigen-3.3.9 [[Baidu]](https://pan.baidu.com/s/19w32kjOIwEGxLNYMZWTKWg) (PWD: ziiw)

## Quick Start
1. Download JDE weights from [[Google]](https://drive.google.com/file/d/1sca65sHMnxY7YJ89FJ6Dg3S3yAjbLdMz/view?usp=sharing) [[Baidu]](https://pan.baidu.com/s/1cCulbPNneIXOpRRjrTgJ4g).
2. Convert the pytorch model to a jit model based on [Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT), or download [[jit_model]](https://pan.baidu.com/s/1d8SEvOYgnwCu4WDjRxev1g) (PWD: tupu) directly. 
```
python cvt2jit.py (based on pytorch-1.4.0)
```
3. Compile source code by VS2017/2019.
4. Run JDETracker.

## Performance
|Model| MOTA | IDF1 | IDS | FP | FN | FPS @Hardware |
|-----|------|------|-----|----|----|----|
|JDE-576x320| 63.7|	63.3|	1307|	6657|	32794| 33.5 @i7-9700K, RTX-2080ti|

## Video Demo
<img src="assets/MOT16-03.gif" width="400"/>   <img src="assets/MOT16-14.gif" width="400"/>

## Reference
[Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)  
[samylee_csdn](https://blog.csdn.net/samylee)
