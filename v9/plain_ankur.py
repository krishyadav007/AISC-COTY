# import streamlit as st
# import tensorflow as tf
import tempfile
import cv2
import numpy as np
import PIL
import keras
import vehicle_detection.pollution_estimator

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
        print("TF retruned : " + returned["results"]["detection_boxes"][0][ii])
        if (returned["results"]["detection_scores"][0][ii]*100) > CONFIDENCE_VALUE:
            xmin = round((returned["results"]["detection_boxes"][0][ii][0] * im_width))
            ymin = round((returned["results"]["detection_boxes"][0][ii][1] * im_height))
            xmax = round((returned["results"]["detection_boxes"][0][ii][2] * im_width))
            ymax = round((returned["results"]["detection_boxes"][0][ii][3] * im_height))
            # ROI = image_cv[ymin:ymin+ymax, xmin:xmin+xmax]
            ROI = image_cv[ymin:ymax, xmin:xmax]
            print(xmin,xmax,ymin, ymax)
            print(np.shape(ROI))
            name = "Image_ROI"+str(ii)+"_v1.png"
            cv2.imwrite(name ,ROI)
            break