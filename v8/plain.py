# import streamlit as st
import tensorflow as tf
import tempfile
import cv2
import numpy as np
import PIL
import keras
import vehicle_detection.pollution_estimator

# PAGE_TITLE = "AI Smoky vehicle detection"

# def video_upload():
#     f = st.file_uploader("Choose a video")
#     if f is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(f.read())
#         cap = cv2.VideoCapture(tfile.name)
#         while(cap.isOpened()):
#             ret, frame = cap.read() 
#             if ret:
#                 PIL_frame = PIL.Image.fromarray(frame)
#                 vehicle_detection(PIL_frame)
#             else:
#                print('no video')
#                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                continue

#         cap.release()

# def image_upload():
#     uploaded_file = st.file_uploader("Choose a image")
#     if uploaded_file is not None:
#         image = PIL.Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#         st.write("")
#         st.write("Processing...")
#         vehicle_detection.pollution_estimator.get_segmentations(np.array(image)[:, :, ::-1].copy() )
#         st.write("Completed...")


# def main():
#     st.set_page_config(page_title=PAGE_TITLE, layout="wide")
#     st.title(PAGE_TITLE)
#     image_upload()
#     # video_upload()
#     # image_path = file_selector_ui()
model = keras.models.load_model("model.h5") # Pollution detection
def preds(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = PIL.ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)[:, :, ::-1].copy()
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    label = np.argmax(prediction)
    return label

if __name__ == "__main__":

    image = cv2.imread("./sub.png")
    im_pil = PIL.Image.fromarray(image)
    (im_width, im_height) = im_pil.size
    image_reshaped = np.array(im_pil.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)
    returned = vehicle_detection.pollution_estimator.get_segmentations( image_reshaped )
    CONFIDENCE_VALUE = 30

    image_cv = cv2.imread("./sub.png")
    print(np.shape(image_cv))

    for ii in range(len(returned["results"]["detection_boxes"][0])):
        if (returned["results"]["detection_scores"][0][ii]*100) > CONFIDENCE_VALUE:
            ymin = round((returned["results"]["detection_boxes"][0][ii][0] * im_height))
            xmin = round((returned["results"]["detection_boxes"][0][ii][1] * im_width))
            ymax = round((returned["results"]["detection_boxes"][0][ii][2] * im_height))
            xmax = round((returned["results"]["detection_boxes"][0][ii][3] * im_width))
            ROI = image_cv[xmin:xmax][ymin:ymax]
            print(xmin,xmax,ymin, ymax)
            print(np.shape(ROI))
            name = "Image_ROI"+str(ii)+".png"
            cv2.imwrite(name ,ROI)

    # print(returned["labeled_image"])
    # print(np.shape(returned["labeled_image"]))
