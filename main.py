
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

image_file = 'hongkong-2669874_960_720.jpg'

import argparse
import colorsys
import os
import imghdr
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

model_path = './model/yolo.h5'
anchors_path = './model/yolo_anchors.txt'
classes_path = './model/coco_classes.txt'

def yolo_boxes_to_corners(box_xy, box_wh):
    """ Chuyển đổi box predictions sang box corners """
    
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):

    box_scores = box_confidence * box_class_probs # box_score sẽ bằng Px nhân với tỉ lệ của 80 class
    box_classes = K.argmax(box_scores, axis=-1) # index tại giá trị lớn nhất ở trục 80 trả về (19,19,5)
    box_class_scores = K.max(box_scores, axis=-1) # giá trị lớn nhất ứng với trục 80 trả về mảng 80 số 
    prediction_mask = box_class_scores >= threshold

    boxes = tf.boolean_mask(boxes, prediction_mask) # sẽ trả về những giá trị ứng với true
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)

    return boxes, scores, classes


def yolo_eval(yolo_outputs,image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width]) # xếp chồng
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    # non max suppression : xử lý giúp boxes không bị chồng chéo
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index) # K.gather để chọn nms index từ boxes
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    
    return boxes, scores, classes

def yolo_head(feats, anchors, num_classes):

    num_anchors = len(anchors) 

    anchors_tensor = K.reshape(K.variable(anchors), [1,1,1, num_anchors, 2])

    conv_dims = K.shape(feats)[1:3] # 19x19

    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2]) # nén giá trị chạy từ 0 -> 1
    box_wh = K.exp(feats[..., 2:4]) # hàm e mũ x 
    box_confidence = K.sigmoid(feats[..., 4:5]) # nén giá trị chạy từ 0 -> 1
    box_class_probs = K.softmax(feats[..., 5:]) # hàm trung bình mũ 

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs

with open(classes_path) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

yolo_model = load_model(model_path)

# Gán mỗi object với một màu để hiển thị
hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Khởi tạo bộ sinh ngẫu nhiên
random.shuffle(colors)  # Các màu tương ứng với các class sẽ được sắp xếp ngẫu nhiên theo bộ sinh
random.seed(None)  # Reset bộ sinh ngẫu nhiên

import matplotlib.pyplot as plt

score_threshold = .3
iou_threshold = .5

sess = K.get_session()

# Kiếm tra xem kích thước của model có tương ứng với input không
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Đưa ra những output tensor cho việc lọc các boxes
# Bọc các lớp xử lý tính toán bằng keras layer
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=score_threshold,
    iou_threshold=iou_threshold)

image = Image.open(image_file)

if is_fixed_size: 
    resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
            # Image.BICUBIC các pixel sau khi resize sẽ tựng động phân bố để có kích thước tốt nhất
    image_data = np.array(resized_image, dtype='float32')
else:
    # Để vượt qua connection + max pooling trong YOLO_v2, đầu vào(width, height) phải là bội số của 32
    new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
    resized_image = image.resize(new_image_size, Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    print(image_data.shape)

image_data /= 255.
image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    #  thêm một tham số làm thứ nguyên đầu tiên và sau đó sử dụng 
    # tất cả các phần tử từ thứ nguyên đầu tiên của mảng ban đầu làm 
    # các phần tử trong thứ nguyên thứ hai của mảng kết quả
    # sẽ có 2 cặp []
    
out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
print('Tìm được {} vật thể ở hình {}'.format(len(out_boxes), image_file))

font = ImageFont.truetype(
        font='FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
for i, c in reversed(list(enumerate(out_classes))):
    predicted_class = class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format(predicted_class, score)

    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    
    draw.rectangle(
                [left , top , right , bottom], width= 4,
                outline=colors[c])
    draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.grid(False)
plt.axis('off')
plt.show()