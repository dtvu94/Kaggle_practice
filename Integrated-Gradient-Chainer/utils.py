import cv2
import numpy as np

import chainer
import chainer.links as clinks
import chainer.functions as F

import os
from os.path import join, abspath, dirname, isfile

# function to get model class object, can add more model definitions by a dict
def create_model(model_name, optional_model_dict=None):
    if optional_model_dict is not None and type(optional_model_dict) is dict:
        a = optional_model_dict.get(model_name, None)
        if a is None:
            model_dict = {
                "vgg16": clinks.VGG16Layers,
                "vgg19": clinks.VGG19Layers,
                "googlenet": clinks.GoogLeNet,
                "resnet50": clinks.ResNet50Layers,
                "resnet101": clinks.ResNet101Layers,
                "resnet152": clinks.ResNet152Layers
            }
            a = model_dict.get(model_name, None)
            if a is None:
                raise ValueError("Cannot find model structure: " + model_name)
            return a()
        else:
            return a()
    else:
        model_dict = {
            "vgg16": clinks.VGG16Layers,
            "vgg19": clinks.VGG19Layers,
            "googlenet": clinks.GoogLeNet,
            "resnet50": clinks.ResNet50Layers,
            "resnet101": clinks.ResNet101Layers,
            "resnet152": clinks.ResNet152Layers
        }
        a = model_dict.get(model_name, None)
        if a is None:
            raise ValueError("Cannot find model structure: " + model_name)
        return a()
    
def read_input_image(image_name, dir_path):
    image_path = join(dir_path, image_name)
    if isfile(image_path) == True:
        img = cv2.imread(image_path)
        # change to RGB color order
        img = img[:, :, (2, 1, 0)]
        return img
    raise ValueError("Cannot read image at path: " + image_path)

def get_predict_and_gradients(inputs, model, label_pred, mean, std):
    gradients = []
    for i, input in enumerate(inputs):
        img = normalize_and_change_shape(input, mean, std)
        y = model(img)
        var_img = chainer.Variable(img)
        prob = model(var_img)['prob']
        # calculate label index - label prediction
        if label_pred is None:
            y = y['prob'].array
            label_pred = y.argmax(axis=1)[0]
        # calculate grad
        prob = prob[:, label_pred]
        gradient = chainer.grad([prob], [var_img])
        # get gradient array
        gradient = gradient[0].array
        gradients.append(gradient)
        print('Done step {}'.format(i))
    gradients = np.array(gradients)
    return label_pred, gradients

def normalize_and_change_shape(img, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]):
    mean = np.array(mean).reshape([1, 1, 3])
    std = np.array(std).reshape([1, 1, 3])
    img = img / 255
    img = (img - mean) / std
    # change to suitable format for chainer (1, 3, x, y)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.array(np.float32(img))
    return img

