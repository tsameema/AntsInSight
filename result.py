from ultralytics import YOLO
import cv2, os
import streamlit as st
from pathlib import Path
from ant_bbox_keypoint_metrics import ANT_DETECTION_METRICS

class PREDICT_MODEL():

    @staticmethod
    def predict_image(image):
        """
        Predict ant bounding boxes and keypoints in an image.

        Args:
            image: Image data for prediction.

        Returns:
            results: Prediction results.
        """

        model = PREDICT_MODEL.choose_model()
        if model is not None:
            return model.predict(image)
        
    @staticmethod
    def calculate_results(results, label_file):
        """
        Calculate IoU and detect ant presence.

        Args:
            results: Prediction results.
            label_file: Path to the ground truth label file.

        Returns:
            is_present: Ant detection status.
            mean_iou: Mean Intersection over Union (IoU).
            error: Mean Squared Error (MSE).
        """
        pred_bbox_norm = results[0].boxes.xywhn.cpu().numpy() #normalized x_y center, width and height
        pred_keypoint = results[0].keypoints.xyn.cpu().numpy()  #normalized pred keypoints
        pred_cls  = results[0].boxes.cls.cpu().numpy()   #pred class

        ispresent, label = ANT_DETECTION_METRICS.find_ant_number_detected(label_file, pred_cls)
        mean_iou, error = ANT_DETECTION_METRICS.calculate_bbox_iou_and_mse(label, pred_bbox_norm, pred_keypoint)

        return ispresent, mean_iou, error

    @staticmethod
    def display_results(results, img, label_file):
        """
        Display and analyze prediction results.

        Args:
            results: Prediction results.
            img: Input image.
            label_file: Path to the ground truth label file.

        Return:
            None
        """
       
        pred_bbox = results[0].boxes.xyxy.cpu().numpy() #x_y center, width and height
        pred_keypoint = results[0].keypoints.xy.cpu().numpy()  #pred keypoints
        for i in range(len(pred_bbox)):
            bbox = pred_bbox[i]
            kp = pred_keypoint[i].astype(int)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
            img = cv2.circle(img, kp[0], 2, (0,255, 0), 1)
            img = cv2.circle(img, kp[1], 2, (0,255, 0), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

        ispresent, mean_iou, error = PREDICT_MODEL.calculate_results(results, label_file)
        
        st.subheader(ispresent)
        st.subheader(mean_iou)
        st.subheader(error)
        st.image(img_rgb)

        [os.remove(file_path) for file_path in Path('tempupload').iterdir() if os.path.isfile(file_path)]
        #shutil.rmtree('runs/pose')

    @staticmethod
    def choose_model():
        """
        Choose a YOLO model based on user input.

        Returns:
            model: Selected YOLO model.
        """

        opt = st.radio('Select model',
                    ("Epochs354_Batch8", "Epochs354_Batch32", "Epochs265_Batch8", "Epochs269_Batch32"))
        if opt == "Epochs354_Batch8":
            return  YOLO('model/best_54_b8.pt')
        elif opt == "Epochs354_Batch32":
            return  YOLO("model/best_54.pt")
        elif opt == "Epochs265_Batch8":
            return YOLO("model/best_265_8.pt")
        elif opt == "Epochs269_Batch32":
            return YOLO("model/best_269_32.pt")
