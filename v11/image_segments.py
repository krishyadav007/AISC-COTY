from yolov5 import YOLOv5
import numpy as np
import glob
import keras
import cv2
from PIL import ImageOps, Image

# Loading the model
model = keras.models.load_model("model.h5")

def machine_classification(img):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)[:, :, ::-1].copy()
    # Normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    # data[0] = normalized_image_array
    data = image_array
    # run the inference
    prediction = model.predict(data)
    label = np.argmax(prediction)
    print(str(prediction))
    # print(image_array, ('Predictions: ' + str(prediction) ), use_column_width=True)
    if label == 0:
        print("The image is of pollution")
        print("Trying to send mail to authorities")
        try:
            genre = input("Do we send the mail (y/n)")
            if genre == None:
                pass
            elif genre == 'Yes':
                # send_mail(text)
                print('You selected yes.')
            else:
                print("You selected no.")
        except Exception as e:
            print(e)
            print("Something went wrong in sending mail")
            print(e)
        else:
            print("Mail sent successfully")
    else:
        print("The image is of vehicle")
    return prediction


model_path = "yolov5x.pt"
device = "cpu"  # or "cuda:0"

img = 'sub.png'
model = YOLOv5(model_path, device)
model.conf = 0.6
object_result = model.predict(img, augment=True)
objects_df = object_result.pandas().xyxy[0]
crops = object_result.crop(save=True)

for name in glob.glob('runs/detect/exp/crops/*/*'):
    print("LOGGING FOR", name)
    img = Image.open(name)
    if img != None:
        # cv2.imread(name)
        img.convert("RGB")
        machine_classification(img)
    # print(name)
    else:
        pass