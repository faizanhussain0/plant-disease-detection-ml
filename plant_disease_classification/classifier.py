#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf

# ===== BASE DIR =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Disable TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()

image_size = 128
num_channels = 3

# ===== PATHS =====
MODEL_DIR = os.path.join(BASE_DIR, 'ckpts')
GRAPH_PATH = os.path.join(MODEL_DIR, 'plants-disease-model.meta')

# 👇 Input image path
IMAGE_PATH = os.path.join(BASE_DIR, 'sample_images', 'leaf.jpg')

# ===== LOAD CLASSES =====
train_dir = os.path.join(BASE_DIR, 'datasets', 'train')
classes = sorted(os.listdir(train_dir))
num_classes = len(classes)

# ===== LOAD MODEL =====
session = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph(GRAPH_PATH)
saver.restore(session, tf.train.latest_checkpoint(MODEL_DIR))

graph = tf.compat.v1.get_default_graph()

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_pred = graph.get_tensor_by_name("y_pred:0")


def classify(image_path=IMAGE_PATH):
    print(f"\nInput image: {image_path}")

    # Check image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) / 255.0

    # Show image (for demo)
    cv2.imshow("Input Leaf Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Prepare input
    x_batch = image.reshape(1, image_size, image_size, num_channels)
    y_dummy = np.zeros((1, num_classes))

    # 🔥 THIS LINE WAS MISSING / BROKEN
    prediction = session.run(
        y_pred,
        feed_dict={x: x_batch, y_true: y_dummy}
    )

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    predicted_class = classes[predicted_index]

    
    print(f"Predicted Disease : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%")
    

if __name__ == "__main__":
    classify()
