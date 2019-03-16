import numpy as np
import cv2
import argparse

import os
from os.path import join, abspath, dirname, exists

from utils import get_predict_and_gradient
from utils import generate_entrie_images
from utils import pre_processing
from utils import create_model
from utils import read_input_image

from integrated_gradients import random_baseline_integrated_gradients

#from visualization import visualize

# prepare paths
BASE_DIR = dirname(abspath(__file__))
RESULT_DIR = join(BASE_DIR, 'results')
IMAGE_DIR = join(BASE_DIR, 'images')

# parse arguments
parser = argparse.ArgumentParser(description='integrated-gradients in Chainer')
parser.add_argument('--model', 
                    type=str, 
                    default='vgg16', 
                    help='the type of network')

parser.add_argument('--image', 
                    type=str, 
                    default='05.jpg', 
                    help='the images name')

if __name__ == '__main__':
    args = parser.parse_args()
    # check result folder
    if exists(RESULT_DIR) == False:
        os.mkdir(RESULT_DIR)
    
    # create models
    model = create_model(args.model)
    print(model)
    # read the image
    img = read_input_image(args.image, IMAGE_DIR)
    
    # mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    
    img = pre_processing(img, mean, std)
    
    # prepare variables for calculating integrated gradient
    steps = 50
    num_random_trials = 10
    clip_above_percentile = 99
    clip_below_percentile = 0
    overlay = True
    mask_mode = True
    outline = True
    
    # calculate the gradient and the label index
    label_index, gradients = get_predict_and_gradient(img, model, None)
    print('label: {}'.format(label_index))
    #print('gradients : {}'.format(gradients))
    """
    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(
                        img, 
                        model, label_index, get_predict_and_gradient,
                        steps, 
                        num_random_trials
                        )
    print(attributions)
    
    img_integrated_gradient_overlay = visualize(
                                            attributions, 
                                            img, 
                                            clip_above_percentile, 
                                            clip_below_percentile,
                                            overlay, 
                                            mask_mode=True
                                            )
    
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                        img_integrated_gradient_overlay)
    
    cv2.imwrite('results/' + args.mode + '/' + args.image, np.uint8(output_img))
    """