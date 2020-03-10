from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import torch
from PIL import Image


class Dataset(Dataset):
    def __init__(self, img_root, datalabels_path, imgH=32, imgW=160, transforms=None):
        super(Dataset, self).__init__()
        self.img_root = img_root
        self.datalabels = self.get_datalabels(datalabels_path)
        self.transforms = transforms
        self.imgH=imgH
        self.imgW=imgW


    def get_datalabels(self, datalabels_path):
        with open(datalabels_path, 'r', encoding='utf-8') as f:
            datalabels = [(line.split(' ')[0], line.split(' ')[-1][:-1]) for line in f.readlines()]
        return datalabels

    def __len__(self):
        return len(self.datalabels)

    def __getitem__(self, index):
        image_name = self.img_root + '/' + self.datalabels[index][0]
        image_label = self.datalabels[index][1]
        image = cv2.imread(image_name)
        # print(self.img_root+'/'+image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        image = cv2.resize(image, (0, 0), fx=self.imgW / w, fy=self.imgH / h, interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (32, self.imgW, 1))).transpose(2, 0, 1)

        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(0.5).div_(0.5)
        return image, image_label


if __name__ == '__main__':
    dataset = Dataset("../data/images/train", "../data/images/train_label.txt",)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    for i_batch, (image, index) in enumerate(dataloader):
        print(image.shape)
