import random
import torch
import torchvision.transforms.functional as TF
import os
import numpy as np
from PIL import Image

from utils import draw_PIL_image, DictAsMember

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
# IMG_HALF = IMG_SIZE / 2

def jitter(x, boxes, config={}):
    """
    brightness_factor: if 1 no operation between 0, 2
    contrast_factor: if 1 no operation between 0, 2
    saturation_factor: f 1 no operation between 0, 2
    hue_factor: if 0 no operation between -0.5, 0.5
    """

    br = config.BR
    con = config.CON
    sat = config.SAT
    hue = config.HUE

    brightness_factor = 1 + random.uniform(-br, br)
    contrast_factor = 1 + random.uniform(-con, con)
    saturation_factor = 1 + random.uniform(-sat, sat)
    hue_factor = random.uniform(-hue, hue)

    x = TF.adjust_brightness(x, brightness_factor=brightness_factor)
    x = TF.adjust_contrast(x, contrast_factor=contrast_factor)
    x = TF.adjust_saturation(x, saturation_factor=saturation_factor)
    x = TF.adjust_hue(x, hue_factor=hue_factor)

    return x, boxes


def vflip(x, boxes):

    x = TF.vflip(x)

    temp_ymin = IMG_SIZE - boxes[:,1]
    temp_ymax = IMG_SIZE - boxes[:,3]
    boxes[:,1] = temp_ymax
    boxes[:,3] = temp_ymin

    return x, boxes


def hflip(x, boxes):

    x = TF.hflip(x)

    temp_xmin = IMG_SIZE - boxes[:,0]
    temp_xmax = IMG_SIZE - boxes[:,2]
    boxes[:,0] = temp_xmax
    boxes[:,2] = temp_xmin

    return x, boxes


def affine(img, boxes, config={}):

    deg = config.DEG
    tr = config.TR * IMG_SIZE
    sc_min, sc_max = config.SC_MIN, config.SC_MAX
    sh = config.SH

    #angle = random.uniform(-deg, deg)
    #translate = (np.round(random.uniform(-tr, tr)),
    #                np.round(random.uniform(-tr, tr)))
    #scale = random.uniform(sc_min, sc_max)
    #shear = random.uniform(-sh, sh)

    #x = TF.affine(img, angle, translate, scale, shear)

    # only implementing for rotate
    rotated_img, boxes = rotate(img, boxes, deg)
    
    return rotated_img, boxes


def rotate(image, boxes, angle):
    """
        Rotates the image and bounding boxes with an angle.

        Code retrieved from:
        https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection
    """
    new_image = image.copy()
    new_boxes = boxes.clone()
    
    #Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w/2
    cy = h/2
    new_image = new_image.rotate(angle, expand=True)

    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    #Get affine matrix
    AffineMatrix = torch.tensor([[alpha, beta, (1-alpha)*cx - beta*cy],
                                 [-beta, alpha, beta*cx + (1-alpha)*cy]])
    
    #Rotation boxes
    box_width = (boxes[:,2] - boxes[:,0]).reshape(-1,1)
    box_height = (boxes[:,3] - boxes[:,1]).reshape(-1,1)
    
    #Get corners for boxes
    x1 = boxes[:,0].reshape(-1,1)
    y1 = boxes[:,1].reshape(-1,1)
    
    x2 = x1 + box_width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + box_height
    
    x4 = boxes[:,2].reshape(-1,1)
    y4 = boxes[:,3].reshape(-1,1)
    
    corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim= 1)
    #corners.reshape(8, 8)    #Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1,2) #Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim= 1) #(Tensors of dimension (4* #objects, 3))
    
    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy
    
    #Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t()).t()
    rotate_corners = rotate_corners.reshape(-1,8)
    
    x_corners = rotate_corners[:,[0,2,4,6]]
    y_corners = rotate_corners[:,[1,3,5,7]]
    
    #Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim= 1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim= 1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim= 1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim= 1)
    y_max = y_max.reshape(-1, 1)
    
    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim= 1)
    
    scale_x = new_image.width / w
    scale_y = new_image.height / h

    #Resize new image to (w, h)
    h, w = image.height, image.width
    new_image = new_image.resize((w, h))
    
    #Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    #new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    #new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    #new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    #new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    
    return new_image, new_boxes

def apply(func, x, targets, prob=0.5, **kwargs):
    if random.random() < prob:
        x, targets = func(x, targets, **kwargs)
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

            # color based
            if self.augment.STATE:
                if self.augment.HORIZONTAL_FLIP.STATE:
                    img, boxes = apply(hflip, img, boxes,
                                       prob=self.augment.HORIZONTAL_FLIP.P)

                if self.augment.VERTICAL_FLIP.STATE:
                    img, boxes = apply(vflip, img, boxes,
                                       prob=self.augment.VERTICAL_FLIP.P)

                if self.augment.JITTER.STATE:
                    img, boxes = apply(jitter, img, boxes,
                                       prob=self.augment.JITTER.P,
                                       config=self.augment.JITTER)

                if self.augment.AFFINE.STATE:
                    img, boxes = apply(affine, img, boxes,
                                       prob=self.augment.AFFINE.P,
                                       config=self.augment.AFFINE)
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
            
            if self.transform is not None:
                img = self.transform(img)

            return img, target, img_name
        
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


if __name__ == "__main__":

    img_path = "data/test_images/000144.jpg"
    img = Image.open(img_path).convert("RGB")
    width = img.width
    height = img.height
    annot = {'boxes': torch.Tensor([[396., 232., 477., 330.],
                                    [135., 191., 438., 333.],
                                    [221., 192., 248., 276.],
                                    [245., 156., 265., 215.],
                                    [  2., 100., 134., 333.],
                                    [166.,  66., 279., 193.],
                                    [351.,   3., 438., 304.],
                                    [344.,   8., 500., 333.]])}
    #annot = {'boxes': torch.Tensor([[539, 1036, 897, 1460]])}
    #annot['boxes'] /= torch.Tensor([[width, height, width, height]])
    #annot['boxes'] *= torch.Tensor([[IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE]])
    #img = img.resize((IMG_SIZE, IMG_SIZE))
    config = DictAsMember({'DEG': 15, 'TR': 0.000001, 'SC_MIN': 0.0000001, 
                           'SC_MAX': 0.000001, 'SH': 0.000001})
    labels = torch.ones(annot['boxes'].shape[0])

    draw_PIL_image(img, annot['boxes'], labels)
    print(annot)
    img, annot = affine(img, annot, config=config)
    print(annot)
    draw_PIL_image(img, annot['boxes'], labels)
