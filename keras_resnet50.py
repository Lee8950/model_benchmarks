import tensorflow as tf
import numpy as np
import os
import platform
import time
import json

resnet = tf.keras.applications.resnet50.ResNet50()

if platform.system() != 'Linux':
    img_prepath = 'E:/imagenet-1k/ILSVRC/Data/CLS-LOC/val'
else:
    img_prepath = '/mnt/e/imagenet-1k/ILSVRC/Data/CLS-LOC/val'
imgs = []

load = 0
for dir in os.listdir(img_prepath):
    i = 0
    for img in os.listdir(f"{img_prepath}/{dir}"):
        tmp = tf.keras.preprocessing.image.load_img(f"{img_prepath}/{dir}/{img}", target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(tmp)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        imgs = imgs + [x]
        load += 1
        i += 1
        if i >= 5:
            break

resnet.summary()

mode = "gpu"

with tf.device(f"/device:{mode}:0"):
    for img in imgs:
        preds = resnet.predict(img)