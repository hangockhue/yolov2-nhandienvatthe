import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

import colorsys
import argparse
import colorsys
import os
import imghdr
import random

import matplotlib.pylot as plt

import YoloDetected as yolo_detected

classes_path = './model/coco_classes.txt'
anchors_path = './model/yolo_anchors.txt'
model_path = './model/yolo.h5'

score_threshold = .3
iou_threshold = .5

# Mở file label
with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

# Mở file anchors size
with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1,2)

yolo_model = load_model(model_path)

hsv_tuples = [(x / len(class_names), 1., 1.)
            for x in range(len(class_names))]

colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))

sess = K.get_session()

model_image_size = yolo_model.layer[0].input_shape[1:3]
is_fixed_size = model_image_size != (None)

yolo_outputs = yolo_detected.yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
box, scores, class_names = yolo_detected.yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=score_threshold,
    iou_threshold=iou_threshold)
