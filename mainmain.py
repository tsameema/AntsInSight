import streamlit as st
import os, warnings
from file_process import FILE_PROCESS
from trained_model_results import MODEL_SELECTION
warnings.filterwarnings("ignore")

def main():
    st.title('Ant Bboxes Keypoints using YOLOv8')

    with st.sidebar:
        opt = st.radio("Select options to view and test model",
                        ("Overview", "Trained Model Results", "Model Prediction")
                    )
    
    save_folder = "tempupload"
    test_label_path = "archive/ant_bbox_keypoint/labels/test"
    path = "result"

    if opt == "Model Prediction":
        handle_model_prediction(save_folder, test_label_path)

    elif opt == "Trained Model Results":
        MODEL_SELECTION.choose_model(path)

    else:
        st.subheader("Overview")
        st.write("This project primarily focuses on the objective of predicting both the bounding boxes and keypoints of ants within images by leveraging the capabilities of the YOLOv8 model. The assessment of predicted results involves the utilization of metrics such as Intersection Over Union (IOU) for bounding boxes and Mean Squared Error (MSE) for keypoints. The pre-trained model has undergone a process of refinement using a customized dataset, during which training was carried out with two distinct batch sizes: 8 and 32.")

def handle_model_prediction(save_folder, test_label_path):
    """
        Uploaded image is being preprocessed, predicted, and plotted using streamlit

    ARGS:
        save_folder (str) : path to save the uploaded file
        test_label_path (str) : path containing test label file

    Returns:
            None
    """
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    file = st.file_uploader("Choose File", type=".png")
    
    if file is not None: 
        fileprocessing = FILE_PROCESS(file, test_label_path, save_folder) 
        imgfile, file_name = fileprocessing.validate_file()
        
        if imgfile is not None:
            fileprocessing.test_model(imgfile, file_name)
        else:
            st.error("Upload Image.png file", icon="🚨")

if __name__ == "__main__":
    main()

