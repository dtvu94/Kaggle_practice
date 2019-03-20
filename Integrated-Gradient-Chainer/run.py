import numpy as np
import imageio
import cv2
import argparse

import os
from os.path import join, abspath, dirname, exists

from utils import get_predict_and_gradients
from utils import create_model
from utils import read_input_image

from integrated_gradients import random_baseline_integrated_gradients

from visualization import visualize

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
    # read the image
    img = read_input_image(args.image, IMAGE_DIR)
    print('image shape after read: {}'.format(img.shape))
    # mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    
    # prepare variables for calculating integrated gradient
    steps = 50
    num_random_trials = 10
    
    # calculate the gradient and the label index
    label_pred, gradients = get_predict_and_gradients([img], model, None, mean, std)
    print('label: {}'.format(label_pred))
    """
    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(
                        img, 
                        model, 
                        label_pred, 
                        get_predict_and_gradients,
                        steps, 
                        num_random_trials,
                        mean, 
                        std)
    np.save('attributions.npy', attributions)
    """
    
    attributions = np.load('attributions.npy')
    print('attributions shape: {}'.format(attributions.shape))
    print('img shape: {}'.format(img.shape))
    
    img_integrated_gradient_overlay = visualize(attributions, 
                                                img, 
                                                clip_above_percentile=95, 
                                                clip_below_percentile=58,
                                                overlay=True)
    
    img_integrated_gradient = visualize(attributions, 
                                        img, 
                                        clip_above_percentile=95, 
                                        clip_below_percentile=58, 
                                        overlay=False)
    
    img_integrated_gradient_overlay = np.uint8(img_integrated_gradient_overlay)
    img_integrated_gradient = np.uint8(img_integrated_gradient)
    imageio.imwrite('./results/img_integrated_gradient_overlay.png', img_integrated_gradient_overlay)
    imageio.imwrite('./results/img_integrated_gradient.png', img_integrated_gradient)
    
    
    #print('attributions: {}'.format(attributions))
    """
    ig_overlay_outline = visualize(attributions, 
                                            img, 
                                            overlay=overlay, 
                                            outlines=outline)
    overlay = False
    ig = visualize(attributions, 
                    img,
                    clip_above_percentile=clip_above_percentile,
                    clip_below_percentile=clip_below_percentile,
                    overlay=overlay)
    ig_overlay_outline = ig_overlay_outline.astype(np.uint8)
    ig = ig.astype(np.uint8)
    imageio.imwrite('ig_overlay_outline.png', ig_overlay_outline)
    imageio.imwrite('ig.png', ig)
    """