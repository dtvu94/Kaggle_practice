# DATASET LINK: https://www.microsoft.com/en-us/download/details.aspx?id=54765

# IMPORT
import numpy as np
import cv2

import os
from os.path import dirname, join, isfile, abspath

import argparse
import json

# PARSE ARGUMENT
parser = argparse.ArgumentParser()
# check if first time running => check the dataset of dog, cat
parser.add_argument(
    '--first-time', 
    type=str, 
    default='No', 
    help='Is this your first time running? (Yes/No)')

args = parser.parse_args()

# FUNCTION
# Load dogs or cats' paths from a text file, return a list of paths
def load_text_info(path):
    if isfile(path) == False:
        raise ValueError(path + " isn't existed!")
    res = []
    with open(path, 'r') as f:
        for line in f:
            tmp = line.replace('\n', '')
            res.append(tmp)
    return res
    
# Save dogs or cats' paths to a text file
def save_image_path(path, list_path):
    with open(path, 'w') as f:
        for line in list_path:
            f.write(line)
            f.write('\n')

# Save mean and standard deviation from both computations into a json file  
def save_results(path, res_dict):
    with open(path, 'w') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

# compute mean, standard deviation using opencv, return a list of mean, std
def compute_mean_std_in_opencv(list_dog_path, list_cat_path):
    list_mean = []
    list_std = []
    print('Start loading dog images in computation function.')
    for i, path in enumerate(list_dog_path):
        img = cv2.imread(path)
        tmp_mean = np.mean(img, axis=tuple(range(img.ndim - 1)))
        tmp_std = np.std(img, axis=tuple(range(img.ndim - 1)))
        list_mean.append(tmp_mean)
        list_std.append(tmp_std)
        if (i % 1000) == 0:
            print('Done {} dog images.'.format(i))
    print('Start loading cat images in computation function.')
    for i, path in enumerate(list_cat_path):
        img = cv2.imread(path)
        tmp_mean = np.mean(img, axis=tuple(range(img.ndim - 1)))
        tmp_std = np.std(img, axis=tuple(range(img.ndim - 1)))
        list_mean.append(tmp_mean)
        list_std.append(tmp_std)
        if (i % 1000) == 0:
            print('Done {} cat images.'.format(i))
    print('Start converting to numpy array.')
    arr_mean = np.asarray(list_mean)
    arr_std = np.asarray(list_std)
    print('arr_img shape: {}'.format(arr_mean.shape))
    list_img = []
    print('Start computing mean and standard deviation.')
    mean = np.mean(arr_mean, axis=tuple(range(arr_mean.ndim - 1)))
    print('mean: {}'.format(mean))
    std = np.std(arr_std, axis=tuple(range(arr_std.ndim - 1)))
    print('std: {}'.format(std))
    return [mean.tolist(), std.tolist()]

# validate dogs, cats image on the first time running, to prevent broken images 
def validate_dog_cat_images_opencv(
    list_dog_path, 
    list_cat_path, 
    dog_path, 
    cat_path):
    print('Start validate {} dog images'.format(len(list_dog_path)))
    tmp_dog_path = []
    checked = False
    for i, path in enumerate(list_dog_path):
        if (i % 1000) == 0:
            print('Done {} dog images.'.format(i))
        img = cv2.imread(path)
        if img is None:
            print('Get error at dog image: {}'.format(path))
            checked = True
        else:
            tmp_dog_path.append(path)
    save_image_path(dog_path, tmp_dog_path)
    print('Start validate {} cat images'.format(len(list_cat_path)))
    tmp_cat_path = []
    for i, path in enumerate(list_cat_path):
        if (i % 1000) == 0:
            print('Done {} cat images.'.format(i))
        img = cv2.imread(path)
        if img is None:
            print('Get error at cat image: {}'.format(path))
            checked = True
        else:
            tmp_cat_path.append(path)
    save_image_path(cat_path, tmp_cat_path)
    return checked
    
# MAIN

BASE_DIR = dirname(abspath(__file__))
PET_DIR = join(BASE_DIR, 'PetImages')
DOG_224_DIR = join(PET_DIR, 'Dog_224')
CAT_224_DIR = join(PET_DIR, 'Cat_224')
DOG_PATH = join(BASE_DIR, 'dogs.txt')
CAT_PATH = join(BASE_DIR, 'cats.txt')
RES_PATH = join(BASE_DIR, 'res.json')

print('BASE_DIR: {}'.format(BASE_DIR))

list_dog_path = load_text_info(DOG_PATH)
list_cat_path = load_text_info(CAT_PATH)

print('Number of cats: {}'.format(len(list_cat_path)))
print('Number of dogs: {}'.format(len(list_dog_path)))

if args.first_time == 'Yes' or args.first_time == 'Y'\
    or args.first_time == 'yes' or args.first_time == 'y':
    # validate all existed images
    checked = validate_dog_cat_images_opencv(
        list_dog_path, 
        list_cat_path, 
        DOG_PATH, 
        CAT_PATH)
    if checked == True:
        list_dog_path = load_text_info(DOG_PATH)
        list_cat_path = load_text_info(CAT_PATH)
    
print('Start computing with opencv')
mean_std = compute_mean_std_in_opencv(list_dog_path, list_cat_path)
res_dict = {
    "mean": mean_std[0],
    "standard_deviation": mean_std[1]
}
save_results(RES_PATH, res_dict)

