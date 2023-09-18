# Ant_bbox_keypoints_prediction_using_yolov8

## Overview
This project is centered around the task of predicting bounding boxes and keypoints of ants present in images using the YOLOv8 model. The pre-trained model has been fine-tuned on a custom dataset, with training conducted using two different batch sizes: 8 and 32. The model has undergone four training iterations, each comprising 500 epochs, with a default patience value to prevent overfitting.

The dataset used for training can be found on Kaggle: Ant 2 Keypoints Dataset on Natural Substrate. However, this dataset had to be transformed into the format required by YOLOv8. The code to achieve this format conversion can be found in the **merge_bbox_keypoints.py** script.

## Model and Deployment
The YOLOv8 model is utilized for this project, with the model file named **'yolov8n-pose.pt'**.

The project is deployed using Streamlit, where two distinct radio boxes are available:

**Results of Trained Model:** This section includes various metrics and visualizations related to the trained model, such as Pose/bbox PR curves, validation prediction results for different batches, confusion matrices, Pose/bbox F1 curves, plots of train/validation loss, mAP50, mAP50-95, precision, recall for both training and validation data, and a dictionary containing all of the aforementioned values for the model.

**Prediction:** In this section, users can upload an image file to be tested with the trained model. The predicted results are displayed, along with metrics such as Intersection Over Union (IOU) for bounding boxes and Mean Squared Error (MSE) for keypoints.

## Usage
To prepare the dataset in the required format for YOLOv8, run merge_bbox_keypoints.py. Once the dataset is ready, execute the main application by running main.py. The main application comprises several steps:

**file_process.py:** This script checks the image file format, performs augmentations, and prepares the image for prediction.

**result.py:** This script predicts the image and generates results, including IOU for bounding boxes and MSE for keypoints. The results are coded in the ant_bbox_keypoint_metrics.py file.

To check the results of training and validation loss, refer to the trained_model_results.py script. The results for each file are saved in the **"results"** folder, which further contains results for each model iteration.

For a detailed explanation of the training process and code, refer to the **ant_keypoint.ipynb** Jupyter Notebook.

## Dependencies
Ensure you have the following dependencies installed:

      !pip install albumentations
      
      !pip install ultralytics==8.0.20
      
      !pip install --upgrade opencv-python matplotlib

## Running the Application
To run the application, execute the following command:

  streamlit run main.py
  
Feel free to explore the project and utilize it for ant bounding box and keypoints prediction using the YOLOv8 model.

  
