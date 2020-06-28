import random
import torch
from torchvision import transforms
import os
from PIL import Image


class SubmitDataset(object):
    
    def __init__(self, image_files_list=[], transform=True):
        self.image_files_list = image_files_list
        self.transform = transform

    def __getitem__(self, idx):
        # load images 
        img_path = self.image_files_list[idx]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1] 

        if self.transform is not None:
            img = self.transform(img)

        return img, width, height

    def __len__(self):
        return len(self.image_files_list)
