import random
import torch
from torchvision import transforms
import os
from PIL import Image

"""
    {'boxes': tensor([ x1, y1, x2, y2
        [252.2484,  70.4748, 314.7751, 156.3727],
        [383.2976,  82.8990, 434.0471, 146.7333],
        [320.9850, 194.7162, 358.0300, 226.2049],
        [314.1328, 280.3999, 349.6788, 311.2460]
    ]), 
    'labels': tensor([1, 1, 1, 1]), 
    'image_id': tensor([2086]), 
    'area': tensor([5370.9160, 3239.5598, 1166.4994, 1096.4570]), 
    'iscrowd': tensor([0, 0, 0, 0])}
"""
IMG_SIZE = 600
IMG_HALF = IMG_SIZE / 2

def jitter(x, targets):
    return x, targets

def vflip(x, targets):
    x = transforms.functional.vflip(x)
    targets['boxes'][:,1] = IMG_HALF - (targets['boxes'][:,1] - IMG_HALF)
    targets['boxes'][:,3] = IMG_HALF - (targets['boxes'][:,3] - IMG_HALF)
    return x, targets

def hflip(x, targets):
    x = transforms.functional.hflip(x)
    targets['boxes'][:,0] = IMG_HALF - (targets['boxes'][:,0] - IMG_HALF)
    targets['boxes'][:,2] = IMG_HALF - (targets['boxes'][:,2] - IMG_HALF)
    return x, targets

def apply(func, x, targets, prob=0.5):
    if random.random() < prob:
        x, targets = func(x, targets)
    return x, targets


class ForeignObjectDataset(object):
    
    def __init__(self, datafolder, datatype='train', transform=True, labels_dict={}, augment=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.labels_dict = labels_dict
        self.image_files_list = [s for s in sorted(os.listdir(datafolder)) if s in labels_dict.keys()]
        self.transform = transform
        self.annotations = [labels_dict[i] for i in self.image_files_list]
        self.augment = augment
            
    def __getitem__(self, idx):
        # load images 
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.datafolder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]  
        
        if self.datatype == 'train':
            annotation = self.labels_dict[img_name]
            
            boxes = []
            
            if type(annotation) == str:
                annotation_list = annotation.split(';')
                for anno in annotation_list:
                    x = []
                    y = []
                
                    anno = anno[2:]
                    anno = anno.split(' ')
                    for i in range(len(anno)):
                        if i % 2 == 0:
                            x.append(float(anno[i]))
                        else:
                            y.append(float(anno[i]))
                        
                    xmin = min(x)/width * 600
                    xmax = max(x)/width * 600
                    ymin = min(y)/height * 600
                    ymax = max(y)/height * 600
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            # color based
            if self.augment.HORIZONTAL_FLIP:
                img, target = apply(hflip, img, target, prob=0.5)

            if self.augment.VERTICAL_FLIP:
                img, target = apply(vflip, img, target, prob=0.5)

            if self.augment.JITTER:
                img, target = apply(jitter, img, target, prob=0.5)
            
            if self.transform is not None:
                img = self.transform(img)
            

            return img, target
        
        if self.datatype == 'dev':
            
            if self.labels_dict[img_name] == '':
                label = 0
            else:
                label = 1
            
            if self.transform is not None:
                img = self.transform(img)

            return img, label, width, height

    def __len__(self):
        return len(self.image_files_list)
