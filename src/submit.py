import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os
import utils
import argparse
from tqdm import tqdm
from PIL import Image

from dataset import SubmitDataset
from utils import DictAsMember
from fasterrcnn_models import _get_detection_model


CONF_WEIGHT = 'FRCNN_R50_FPN_2x_DA.pt'
#CONF_WEIGHT = 'FRCNN_R152_FPN_2x_DA.pt'
CONF_NMS = 0.05
CONF_DET = 0.25
#CONF_MODEL_NAME = 'resnet152'
CONF_MODEL_NAME = 'resnet50'
CONF_MODEL_NCLASS = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CXR Object Localization')
    parser.add_argument('image_path', type=str, metavar='IMAGE_PATH',
                        help='')
    parser.add_argument('predictions_classification', type=str, metavar='PREDICTIONS_CLASSIFICATION',
                        help='')
    parser.add_argument('predictions_localization', type=str, metavar='PREDICTIONS_LOCALIZATION',
                        help='')
    args = parser.parse_args()
    # yaml_path = CONF_YAML
    # with open(yaml_path, 'r') as f:
    #     exp_args = DictAsMember(yaml.safe_load(f))

    model = _get_detection_model(CONF_MODEL_NCLASS,
                                 CONF_MODEL_NAME,
                                 box_nms_thresh=CONF_NMS,
                                 box_score_thresh=CONF_DET)

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([transforms.Resize((600, 600)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=img_mean,
                                                               std=img_std)]
                                         )

    image_files_list = [line for line in open(args.image_path, 'r').read().splitlines(keepends=False)]
    dataset = SubmitDataset(image_files_list=image_files_list,
                                transform=data_transforms)

    data_loader_val = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False, num_workers=4)
                                 #collate_fn=utils.collate_fn)

    #device = torch.device('cuda:0')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model.load_state_dict(torch.load(CONF_WEIGHT))
    model.to(device)

    model.eval()

    preds = []
    locs = []

    idx = 0
    for image, width, height in tqdm(data_loader_val):
        
        image = image.to(device)
        outputs = model(image)
        
        center_points = []
        center_points_preds = []
        
        if len(outputs[-1]['boxes']) == 0:
            preds.append(0)
            center_points.append([])
            center_points_preds.append('')
            locs.append('')
        else:
            preds.append(torch.max(outputs[-1]['scores']).tolist())
            
            new_output_index = torch.where((outputs[-1]['scores']>0.1))
            new_boxes = outputs[-1]['boxes'][new_output_index]
            new_scores = outputs[-1]['scores'][new_output_index]
            
            print(new_scores)
            print(new_boxes)
            utils.draw_PIL_image(
                Image.open(dataset.image_files_list[idx]).convert("RGB").resize((600,600)), 
                new_boxes, None)

            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                center_x = (new_box[0] + new_box[2])/2
                center_y = (new_box[1] + new_box[3])/2
                center_points.append([center_x/600 * width[-1],center_y/600 * height[-1]])
            center_points_preds += new_scores.tolist()
            
            line = ''
            for i in range(len(new_boxes)):
                if i == len(new_boxes)-1:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0].item()) + ' ' + str(center_points[i][1].item())
                else:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0].item()) + ' ' + str(center_points[i][1].item()) +';'
            locs.append(line)
        idx += 1

    cls_res = pd.DataFrame({'image_name': dataset.image_files_list,
                            'prediction': preds})
    cls_res.to_csv(args.predictions_classification, columns=['image_name', 'prediction'],
                   sep=',', index=None)
    print('classification.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset.image_files_list,
                            'prediction': locs})
    loc_res.to_csv(args.predictions_localization, columns=['image_name', 'prediction'],
                   sep=',', index=None)
    print('localization.csv generated.')

