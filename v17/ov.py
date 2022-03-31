import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork 
from openvino.inference_engine import IECore

def load_IR_to_IE(model_xml):
    ### Load the Inference Engine API
    plugin = IECore()
    ### Loading the IR files to IENetwork class
    model_bin = model_xml[:-3]+"bin" 
    print(model_bin)
    # this way pretty old and deprecated, please use plugin.read_network instead
    network = plugin.read_network(model=model_xml, weights=model_bin)
    ### Loading the network
    executable_net = plugin.load_network(network,"CPU")
    print("Network succesfully loaded into the Inference Engine")
    return executable_net

def synchronous_inference(executable_net, image):
    ### Get the input blob for the inference request
    # also old API, now inputs are removed, please use input_info instead
    input_blob = next(iter(executable_net.input_info))
    ### Perform Synchronous Inference
    result = executable_net.infer(inputs = {input_blob: image})
    return result

en = load_IR_to_IE('intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml')

# these actions are wrong, opnvino expects numpy array as input tensor and model expects data in [0, 255] range
#from torchvision import transforms
#tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
#img_tensor = tran(resized)

# correct preprocessing steps transose image layout from HWC to CHW and add batch dim

def DrawBoundingBoxes(predictions, image, conf=0):
    canvas = image.copy()                             # copy instead of modifying the original image
    predictions_1 = predictions[0][0]                 # subset dataframe
    print(predictions_1.ndim)
    confidence = predictions_1[:,2]                   # getting conf value [image_id, label, conf, x_min, y_min, x_max, y_max]
    topresults = predictions_1[(confidence>conf)]     # choosing only predictions with conf value bigger than treshold
    (h,w) = canvas.shape[:2]                        # 
    for detection in topresults:
        box = detection[3:7] * np.array([w, h, w, h]) # determine box location
        (xmin, ymin, xmax, ymax) = box.astype("int") # assign box location value to xmin, ymin, xmax, ymax

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)  # make a rectangle
        cv2.putText(canvas, str(round(detection[2]*100,1))+"%", (xmin, ymin), # include text
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    cv2.putText(canvas, str(len(topresults))+" vehicle license plates detected", (50,50), # include text
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    return canvas
def procceess(cv2_img):
    image = cv2_img
    print(image.shape)
    resized = cv2.resize(image, (300,300))

    img_tensor = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0)
    res = synchronous_inference(en, img_tensor)
    # cv2_imshow(cv2.imread('/content/car4.jpg'))

    print(res.get('DetectionOutput_')[np.ndarray.nonzero(res.get('DetectionOutput_'))])
    print(res.get('DetectionOutput_'))
    result = np.asarray(res["DetectionOutput_"], dtype=np.float32)
    np.shape(result[0][0][2])
    return DrawBoundingBoxes(result,image)