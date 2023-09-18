import streamlit as st
import os


class MODEL_SELECTION():

    @staticmethod
    def choose_model(path):
        """
        Select any model and display its results

        ARGS:
            path (str) : path to the required model results

        RETURN:
            streamlit : Return results on the webpage
        """
        opt = st.radio('Select model',
                    ("", "Epochs354_Batch8", "Epochs354_Batch32", "Epochs265_Batch8", "Epochs269_Batch32"))
        if opt == "Epochs354_Batch8":
            result_path = os.path.join(path, "best_54_b8")
            return  MODEL_SELECTION.display_model_results(result_path)
        elif opt == "Epochs354_Batch32":
            result_path = os.path.join(path, "best_54")
            return  MODEL_SELECTION.display_model_results(result_path)
        elif opt == "Epochs265_Batch8":
            result_path = os.path.join(path, "best_265_8")
            return  MODEL_SELECTION.display_model_results(result_path)
        elif opt == "Epochs269_Batch32":
            result_path = os.path.join(path, "best_269_32")
            return  MODEL_SELECTION.display_model_results(result_path) 

    @staticmethod
    def display_model_results(folder):
        """
        Display required trained model results on streamlit

        ARGS:
            folder (str) : A path of selected trained model results
        """
        file = os.listdir(folder)

        #Display all the results in the image form
        for f in file:
            if f.endswith('jpg') or f.endswith('.png'):
                st.subheader(f.split('.')[0])
                st.image(os.path.join(folder, f))

        #Display the trained model metrics
        with open(os.path.join(folder, 'metric.txt'), 'r+') as f:
            metric = f.readlines()
        f.close
        st.subheader('Trained Model Metrics')
        st.write(metric)


