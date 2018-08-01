import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops       # was from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/NTran/OneDrive - g.hmc.edu/CS Pixel/TFc NeuralNet/models/research/object_detection/data/mscoco_label_map.pbtxt'

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


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loads the labels to match with detected objects
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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
        break

# if VideoCapture(feed) doesn't work, manually try -1, 0, 1, 2, 3 (if none of those work, the webcam's not supported!)
cam = cv2.VideoCapture(feed)
    
with detection_graph.as_default():
  with tf.Session(graph = detection_graph) as sess:
    while True:
        ret, image = cam.read() 

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
        # Actual detection.
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
  
        cv2.imshow('Webcam',image)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cam.release()

