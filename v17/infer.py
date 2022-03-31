import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import cv2
import numpy as np
from six import BytesIO
import license_plate as lp
from PIL import Image, ImageDraw, ImageFont
import re
import tensorflow as tf
import ov

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# def load_resize_cv2(image):
  # return image.reshape(
  #     (im_height, im_width, 3)).astype(np.uint8)
label_map_pbtxt_fname = './train/Smoke_label_map.pbtxt'
##Change chosen model to deploy different models available in the TF2 object detection zoo
MODELS_CONFIG = {
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d1': {
        'model_name': 'efficientdet_d1_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d1_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d1_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
    'efficientdet-d2': {
        'model_name': 'efficientdet_d2_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d2_768x768_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d2_coco17_tpu-32.tar.gz',
        'batch_size': 16
    },
        'efficientdet-d3': {
        'model_name': 'efficientdet_d3_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d3_896x896_coco17_tpu-32.config',
        'pretrained_checkpoint': 'efficientdet_d3_coco17_tpu-32.tar.gz',
        'batch_size': 16
    }
}
#in this tutorial we implement the lightweight, smallest state of the art efficientdet model
#if you want to scale up tot larger efficientdet models you will likely need more compute!
chosen_model = 'efficientdet-d0'
num_steps = 10000 #The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing. 
num_eval_steps = 500 #Perform evaluation after so many steps

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
batch_size = MODELS_CONFIG[chosen_model]['batch_size'] #if you can fit a large batch in memory, it may speed up your training
pipeline_file = './models/research/deploy/pipeline_file.config'
model_dir = './training/'

#recover our saved model
pipeline_config = pipeline_file
#generally you want to put the last ckpt from training in here
model_dir = './Smoke-Detection-using-Tensorflow-2.2/fine_tuned_model/checkpoint/ckpt-0'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)
#prepare
pipeline_fname = './models/research/deploy/' + base_pipeline_file
fine_tune_checkpoint = './models/research/deploy/' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

num_classes = get_num_classes(label_map_pbtxt_fname)
#map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join('./Smoke-Detection-using-Tensorflow-2.2/fine_tuned_model/checkpoint/ckpt-0'))

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

def count_boxes(dection_scores, confidence_treshold = 40):
  cnt = 0
  for dection_score in dection_scores:
    if((dection_score*100) > confidence_treshold):
      cnt+=1
  return cnt

def predict_infer(img_cv):
  # TEST_IMAGE_PATHS = "./smoke.jpg"
  image_np = img_cv
  # image_np = load_image_into_numpy_array("smoky_test.jpg")
  input_tensor = tf.convert_to_tensor(
      np.expand_dims(image_np, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)
  no_of_boxes = count_boxes(detections['detection_scores'][0])
  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
  )
  print(no_of_boxes)

  print("IMAGE CHECKED")
  # cv2.imwrite("output.jpg",image_np_with_detections)
  return (no_of_boxes, image_np_with_detections)

def smoke_detection_json(frame, time_id):
    no_of_boxes, result_image = predict_infer(frame)
    if no_of_boxes > 0: 
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        try:
          lp_no = lp.license_plate_pipeline(result_image)
          lp_no = lp_no.lower()
          lp_no = re.sub(r'[^\w]', '', lp_no)
          lp_no = lp_no.strip()
          vinoed_img = ov.procceess(result_image)
          image_PIL = Image.fromarray(result_image)

          with open("static/logs/"+time_id+".txt", "a") as fo:
            buffered = BytesIO()
            image_PIL.save(buffered, format="PNG")
            buffered.seek(0)
            img_byte = buffered.getvalue()
            img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
            fo.write('["'+lp_no + '","'+img_str+'"],')
            # shutil.copyfile("static/logs/"+time_id+".txt", "static/logs/c_"+time_id+".txt")
        except:
            print("SOME ERROR OCCURED, SKIPPING THE FRAME")
    
            # # if lp_no not in LIST_OF_LP:
            #     LIST_OF_LP.append(lp_no)
            #     st.image(result_image, caption="Detected image", width=300)
            #     st.write('License plate : ' + lp_no)
            #     st.write('Do we send mail')
            #     if st.button('Yes', key=lp_no+"y"):
            #         mailer.send_mail(lp_no)
            #         st.write("Mail sent")
            #     if st.button('No', key=lp_no+"n"):
            #         st.write("Mail cancelled")
def proccess(file_name, time_id):
  cap = cv2.VideoCapture("static/uploads/"+file_name)
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  with open("static/logs/"+time_id+".txt", "x") as fo:
    fo.write('{"data":[')
  # print("THE FILE SIZE IS " + str(length))
  cnt = 0
  while(cap.isOpened()):
      ret, frame = cap.read() 
      if ret:
          smoke_detection_json(frame, time_id=time_id)
      else:
          print('no video')
          cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
          break
      print( "The progress is : ", cnt, "/",length )
      if cnt > 10:
        break
      cnt += 1
  cap.release()
  with open("static/logs/"+time_id+".txt", "a") as fo:
    fo.write('["EOF","EOF"]]}')
    # fo.write(']}')
    print("IT IS RUNING OR NOT")
  return "PROCCESSING COMPLETED"

def results(time_id):
  with open("static/logs/"+time_id+".txt", "r") as fo:
    return fo.read()
