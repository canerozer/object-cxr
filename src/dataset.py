import random
import torch
from torchvision import transforms
import os
from PIL import Image


class SubmitDataset(object):
    
    def __init__(self, datafolder, transform=True):
        self.datafolder = datafolder
        self.image_files_list = [s for s in sorted(os.listdir(datafolder))]
        self.transform = transform

    def __getitem__(self, idx):
        # load images 
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_files_list)
