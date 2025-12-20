#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import cv2
import numpy as np
import glob

from sklearn.utils import shuffle
from dataset import DataSet


def find_images(path):
    directory = os.path.join('plant_disease_classification', path)
    extension_list = ['jpg', 'png', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP']
    find_options = '-iname "*.{0}"'.format(extension_list[0])
    for ext in extension_list[1:]:
        find_options += ' -o -iname "*.{0}"'.format(ext)

    process = subprocess.Popen(
        ['find -L {} {}'.format(directory, find_options)],
        stdout=subprocess.PIPE,
        shell=True
    )

    image_paths = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        image_paths.append(line.strip())

    return image_paths


def load_train_data(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    class_array = []

    extension_list = ('*.jpg', '*.JPG')
    MAX_IMAGES = 1000  # RAM safety limit

    print('Going to read training images')

    for fields in classes:
        index = classes.index(fields)
        print(f'Now going to read {fields} files (Index: {index})')

        for extension in extension_list:
            path = os.path.join(train_path, fields, extension)
            files = glob.glob(path)

            for fl in files:
                if len(images) >= MAX_IMAGES:
                    break

                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size))
                image = image.astype(np.float32) / 255.0

                images.append(image)

                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)

                img_names.append(os.path.basename(fl))
                class_array.append(fields)

            if len(images) >= MAX_IMAGES:
                break

        if len(images) >= MAX_IMAGES:
            break

    return (
        np.array(images),
        np.array(labels),
        np.array(img_names),
        np.array(class_array)
    )


def read_train_sets(train_path, image_size, classes, validation_size):
    images, labels, img_names, class_array = load_train_data(
        train_path, image_size, classes
    )

    images, labels, img_names, class_array = shuffle(
        images, labels, img_names, class_array
    )

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    data_set = DataSet()

    data_set.train = DataSet(
        images[validation_size:],
        labels[validation_size:],
        img_names[validation_size:],
        class_array[validation_size:]
    )

    data_set.valid = DataSet(
        images[:validation_size],
        labels[:validation_size],
        img_names[:validation_size],
        class_array[:validation_size]
    )

    return data_set
