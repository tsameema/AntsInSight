# Ant_bbox_keypoints_prediction_using_yolov8

## Overview
This project is centered around the task of predicting bounding boxes and keypoints of ants present in images using the YOLOv8 model. The pre-trained model has been fine-tuned on a custom dataset, with training conducted using two different batch sizes: 8 and 32. The model has undergone four training iterations, each comprising 500 epochs, with a default patience value to prevent overfitting.

The dataset used for training can be found on Kaggle: Ant 2 Keypoints Dataset on Natural Substrate. However, this dataset had to be transformed into the format required by YOLOv8. The code to achieve this format conversion can be found in the **merge_bbox_keypoints.py** script.
