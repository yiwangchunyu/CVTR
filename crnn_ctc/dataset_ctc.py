from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor


class Dataset(Dataset):
    def __init__(self, img_root, datalabels_path, imgH=32, imgW=160, std_mean=(0.5,0.5), direction='horizontal',transforms=None):
        super(Dataset, self).__init__()
        self.img_root = img_root
        self.datalabels = self.get_datalabels(datalabels_path)
        self.transforms = transforms
        self.imgH=imgH
        self.imgW=imgW
        self.std=std_mean[0]
        self.mean = std_mean[1]
        self.rot90 = False
        if direction=='vertical':
            self.rot90=True
        self.toTensor=ToTensor()


    def get_datalabels(self, datalabels_path):
        with open(datalabels_path, 'r', encoding='utf-8') as f:
            datalabels = [(line.split(' ')[0], line.split(' ')[-1][:-1]) for line in f.readlines()]
        return datalabels

    def __len__(self):
        return len(self.datalabels)

    def __getitem__(self, index):
        image_name = self.img_root + '/' + self.datalabels[index][0]
        image_label = self.datalabels[index][1]
        image = Image.open(image_name).convert('L')
        if self.rot90:
            image=image.transpose(Image.ROTATE_90)
        image=image.resize((self.imgW,self.imgH), Image.BILINEAR)
        image = self.toTensor(image)
        image.sub_(self.mean).div_(self.std)
        return image, image_label


if __name__ == '__main__':
    dataset = Dataset("../data/images/train", "../data/images/train_label.txt",)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    for i_batch, (image, index) in enumerate(dataloader):
        print(image.shape)
        print(index)
