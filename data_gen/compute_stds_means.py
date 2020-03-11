import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_std_mean(root,path, imgW, imgH):
    print('computing mean and std...')
    with open(path,encoding='utf-8') as f:
        datalabels=f.readlines()
    num_images = len(datalabels)
    std_mean={}

    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    for datalabel in tqdm(datalabels):
        path=root+'/'+datalabel.split()[0]
        img=Image.open(path).convert('RGB')
        img=np.array(img)
        img=img/255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
    means.reverse()
    stdevs.reverse()
    means = np.asarray(means) / num_images
    stdevs = np.asarray(stdevs) / num_images
    std_mean['RGB']={'std':stdevs.tolist(),'mean':means.tolist()}


    mean=0
    std=0
    for datalabel in tqdm(datalabels):
        path = root + '/' + datalabel.split()[0]
        img = Image.open(path).convert('L')
        img = np.array(img)
        img=img/255.0
        mean+=img[:,:].mean()
        std+=img[:,:].std()

    mean=mean/num_images
    std=std/num_images
    std_mean['L']={'std':std,'mean':mean}

    print(std_mean)
    json.dump(std_mean, open('../data/images/desc/mean_std.json', 'w', encoding='utf-8'))
    return std_mean


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/images/train', help='')
    parser.add_argument('--path', type=str, default='../data/images/train_label.txt', help='')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
    arg = parser.parse_args()
    compute_std_mean(arg.root, arg.path, arg.imgW, arg.imgH)