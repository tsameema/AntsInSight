    
import numpy as np
from sklearn.metrics import mean_squared_error as mse

class ANT_DETECTION_METRICS():
    
    @staticmethod
    def find_ant_number_detected(label_file, pred_cls):
        """
        Find the number of ants detected compared to the ground truth labels.

        Args:
            label_file_path (str): Path to the ground truth label file.
            predicted_classes (list): List of predicted class labels.

        Returns:
            tuple: A tuple containing a message about ant detection and the actual ground truth labels.
        """
        with open(label_file, 'r') as f:
            actual = f.readlines()

        ispresent = len(pred_cls)-len(actual)
        if ispresent==0:
            return 'All Ants predicted', actual
        return (f'{abs(ispresent)} ant present but not predicted ', actual) if ispresent<0 else (f'{abs(ispresent)} ant not present but predicted', actual)

    @staticmethod
    def calculate_bbox_iou_and_mse(actual_labels, predicted_bboxes, predicted_keypoints):
        """
        Calculate the mean Intersection over Union (IoU) and Mean Squared Error (MSE) between 
        actual and predicted labels.

        Args:
            actual_labels (list of str): Ground truth label data.
            predicted_bboxes (list of list): Predicted bounding boxes coordinates.
            predicted_keypoints (list of list): Predicted keypoints coordinates.

        Returns:
            str: A formatted string containing the BBox Mean IOU.
            str: A formatted string containing the Mean Squared Error (MSE).
        """
        if not predicted_bboxes.any():
            return 'BBox Mean IOU: 0.0', 'Mean Squared Error: 0.0'

        ious, mse_error = [], 0

        for gnd in actual_labels:
            #Extract bboxes and keypoints from ground truth labels
            gnd_values = list(map(float, gnd.strip().split(' ')))
            #Calculating IoU
            iou_values = [ANT_DETECTION_METRICS.calculate_iou(pb, gnd_values[1:5]) for pb in predicted_bboxes]
            #Find predicted bbox having max IoU with truth labels
            max_iou_index = np.argmax(iou_values)
            ious.append(np.max(iou_values))
            #Calculate MSE between actual and predicted keypoints
            mse_error += mse(predicted_keypoints[max_iou_index], (gnd_values[5:7], gnd_values[7:]))
        #Calculate mean IoU
        mean_iou = np.mean(ious) * 100
        mean_squared_error = mse_error / len(predicted_bboxes)

        return f'BBox Mean IOU: {mean_iou:.2f}', f'Mean Squared Error: {mean_squared_error:.8f}'

    @staticmethod
    def calculate_iou(pred, gnd):
        """
        Calculate Intersection over union (IoU) between predicted and actual labels

        Args:
            pred (list) : Predicted Bounding boxes Coordinates [x, y, width, height]
            gnd (list)  : Ground Truth Bounding boxes Coordinates [x, y, width, height]

        Return:
            float : Intersection over Union (IoU) score
        """

        x1 = max(gnd[0], pred[0])
        y1 = max(gnd[1], pred[1])
        x2 = min(gnd[0] + gnd[2], pred[0] + pred[2])
        y2 = min(gnd[1] + gnd[3], pred[1] + pred[3])
        
        intersect  = max(0, x2 - x1) * max(0, y2 - y1)

        area_gnd  = gnd[2] * gnd[3]
        area_pred = pred[2] * pred[3]
        union = area_gnd + area_pred - intersect

        return intersect/union if union > 0 else 0.0
