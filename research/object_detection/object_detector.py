# Import required packages - most are available by default with Python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

# This import line prevents an annoying matplotlib backend error - this line is taken from object_detection/utils/visualization_utils.py
import matplotlib; matplotlib.use('Agg') # pylint: disable=multiple-statements

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops       # changed from 'from object_detection.utils import ops as utils_ops'

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

################# MODEL PREPARATION ###################

''' Variables -
    Any model exported using the export_inference_graph.py tool can be loaded here simply by 
    changing PATH_TO_CKPT to point to a new .pb file.

    By default we use an "SSD with Mobilenet" model here. See the detection model zoo for a list of other 
    models that can be run out-of-the-box with varying speeds and accuracies.
    Model Zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
'''

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'INSERT PATH TO .PBTXT LABEL FILE CORRESPONDING TO LOADED MODEL'

# Change number of classes to match number in .pbtxt file
NUM_CLASSES = 90

# Checks whether the model is already downloaded, if not, downloads it
if not os.path.exists("./" + MODEL_NAME):
  print("Downloading base model...")
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

# Loads a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

################# LABEL PREPARATION ###################

''' Label Map -
    Label maps map indices to category names, so that when our convolution network predicts 5, we know that 
    this corresponds to airplane. Here we use internal utility functions, but anything that returns a 
    dictionary mapping integers to appropriate string labels would be fine
'''

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

################# CAMERA PREPARATION ###################

# Reads in source for cam feed
# IF THE SYSTEM ASKS YOU TO CHOOSE A FEED SOURCE, PRESS THE ESC KEY INSTEAD
feed = 0
for src in range(-1, 4):
    temp = cv2.VideoCapture(src)
    ret_val, testImg = temp.read()
    if testImg is None:
        pass
    else:
        feed = src
        temp.release()
        break

# if VideoCapture(feed) doesn't work, manually try -1, 0, 1, 2, 3 (if none of those work, 
# the webcam's not supported!)
cam = cv2.VideoCapture(feed)
    
with detection_graph.as_default():
  # Starts a Tensorflow session
  with tf.Session(graph = detection_graph) as sess:
    # Starts the loop of reading in camera frames
    while True:
        # Reads in each frame separately as individual images
        ret, image = cam.read() 

        # If feed cuts out, loop will restart until another frame is passed in
        if image is None:
          continue

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)
        # Actual detection
        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

        # Display the processed frame
        cv2.imshow('Webcam', image)

        # Pressing ESC will close the window - do not click the window's 'X' to close it
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()




'C:/Users/NTran/OneDrive - g.hmc.edu/CS Pixel/TFc NeuralNet/models/research/object_detection/data/mscoco_label_map.pbtxt'