import streamlit as st
import tensorflow as tf
import tempfile
import cv2
import numpy as np
import PIL 
import vehicle_detection.pollution_estimator

PAGE_TITLE = "AI Smoky vehicle detection"

def video_upload():
    f = st.file_uploader("Choose a video")
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        cap = cv2.VideoCapture(tfile.name)
        while(cap.isOpened()):
            ret, frame = cap.read() 
            if ret:
                PIL_frame = PIL.Image.fromarray(frame)
                vehicle_detection(PIL_frame)
            else:
               print('no video')
               cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
               continue

        cap.release()

def image_upload():
    uploaded_file = st.file_uploader("Choose a image")
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Processing...")
        vehicle_detection.pollution_estimator.get_segmentations(np.array(image)[:, :, ::-1].copy() )
        st.write("Completed...")


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    image_upload()
    # video_upload()
    # image_path = file_selector_ui()

if __name__ == "__main__":
    main()