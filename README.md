# YoloFaster_Rcnn-Detection

This project was developed by António Pedro Silva Matos for object detection using the YOLO (You Only Look Once) and Faster R-CNN (Regions with Convolutional Neural Networks) models. The goal is to provide a unified platform to explore, test, and compare the performance of these models across diverse datasets and scenarios. 

## Summary

1. [**Introduction**](#introduction)
2. [**Overview**](#overview)
3. [**Functionalities**](#functionalities)
4. [**Configuration**](#configuration)
5. [**Requirements**](#requirements)
6. [**Configuration**](#configuration)
7. [**Preperation**](#preperation)
   - [**Download**](#download)
   - [**Installation**](#installation)
8. [**Contribution**](#contribution)
9. [**License**](#license)

## Introduction

This repository presents an innovative platform for object detection, combining two of the most advanced models in the field: YOLO and Faster R-CNN. Developed to provide a detailed comparison between the two approaches, the project aims to help researchers and developers explore and evaluate the effectiveness of each model in diverse scenarios.

## Overview

Object detection is a crucial area of ​​computer vision, with applications in surveillance, autonomous driving, video analysis, and more. In this repository you will find:

  -> YOLO and Faster R-CNN implementations using popular frameworks such as TensorFlow and PyTorch.
  
  -> Scripts to train and evaluate both models on different datasets.
  
  -> Detailed performance comparisons, including metrics such as precision, recall, and inference time.

  -> Guides to help understand and replicate results.
  

## Functionalities

 -> YOLO (You Only Look Once): An object detection model known for its speed and accuracy.

 -> Faster R-CNN: A detection model that uses region proposals to improve detection accuracy.

 -> Performance Comparison: Scripts to evaluate and compare the performance of models in terms of different metrics.

 -> Datasets: Support for multiple popular datasets such as COCO, Pascal, VOC, and others.

## Requirements

 - Python 3.7 or higher
 
 - TensorFlow 2.x or PyTorch 1.x

 - Additional libraries listed in requirements.txt

## Configuration

This project uses `Yolov4`, that includes the files `yolov4.cfg`, `yolov4.weights` and `coco.names`

Before execute the code, verify if the files are present on directory after downloaded the repository

- `Yolov4.cfg`: Model configuration
- `yolov4.weights`: Model weights
- `coco.names`: File with the names of the objects that the model was trained to detect

# Preperation

### Download

Before downloading the repository, we need to download the model settings and weights so that we can use the program itself. I'm using the normal YoloV4, but if your PC doens't have to much memory. I recomend to use tiny YOLO, where will the links to download be, however in the code they will have to modify the file name so that it can run correctly.

 - YoloV4 [weights](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights&ved=2ahUKEwj3j5GIz-eGAxUjTaQEHY0uCSoQFnoECBkQAQ&usg=AOvVaw30if4joxtTaS8DAh12vYQ4)
 - YoloV4 [config](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
 - YoloV4 [classes](https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.names)

For tiny Yolo:
- YoloV4-tiny [config](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)
- YoloV4-tiny [weights](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights&ved=2ahUKEwin8bL5z-eGAxV_hP0HHaS3B3IQFnoECBUQAQ&usg=AOvVaw0mQ6LZDwchkF37sFuwpNSi)

### Installation

1. Clone repository:
   ```
      git clone https://github.com/seu-usuario/YoloFaster_Rcnn-Detection.git
   ```

2. Navigate to the project directory:
   ```
      cd YoloFaster_Rcnn-Detection
   ```

3. Install dependencies:
   ```
      pip install -r requirements.txt
   ```

## Contribution

Feel free to open issues and submit pull requests. Contributions are welcome!

We consider exploring the Darknet YOLO ecosystem for valuable insights and testing different dataset versions and models. Our goal is to improve object detection using the official [Darknet repository](https://github.com/AlexeyAB/darknet). We plan to create a personalized dataset with relevant objects to improve the mobility of the visually impaired, aiming to significantly increase detection accuracy.

For this code we utilize a flexx2 depth camera from [Pmdtec](https://3d.pmdtec.com/en/3d-cameras/flexx2/)

# License

YoloFaster_Rcnn-Detection itself is released under the MIT License (refer to the LICENSE file for details). Portions of the code are borrowed and given by [pmdtechnologies](https://github.com/pmdtechnologies/SampleYOLO) and from their [RoyaleSDK](https://pmdtec.com/en/download-sdk/)
