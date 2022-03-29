# git clone "https://github.com/abg3/Smoke-Detection-using-Tensorflow-2.2"
# curl -L "https://app.roboflow.ai/ds/EVwoZwzA30?key=6OawcH9tOw" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

mkdir "./models/research/deploy/"
cd "./models/research/deploy/"
wget 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
tar -xf + "efficientdet_d0_coco17_tpu-32.tar.gz"

wget 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config'