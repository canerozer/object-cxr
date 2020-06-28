import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import utils
import argparse
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

from dataset import ForeignObjectDataset
from engine import train_one_epoch
from utils import DictAsMember
from fasterrcnn_models import _get_detection_model


CONF_YAML = "configs/faster_rcnn"
CONF_NMS = 0.05
CONF_DET = 0.25
CONF_WEIGHT




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CXR Object Localization')
    parser.add_argument('--yaml', type=str, metavar='YAML',
                        default="configs/faster_rcnn",
                        help='Enter the path for the YAML config')
    parser.add_argument('--nms', type=float, metavar='NMS',
                        default=0.05,
                        help="Enter the NMS threshold for Faster R-CNN")
    parser.add_argument('--det', type=float, metavar='DET',
                        default=0.25,
                        help="Enter the detection threshold for Faster R-CNN")
    parser.add_argument('--weight', type=str, metavar='WEIGHT',
                        default=None,
                        help="Enter the model weights if overwrite is required")
    args = parser.parse_args()

    yaml_path = args.yaml
    with open(yaml_path, 'r') as f:
        exp_args = DictAsMember(yaml.safe_load(f))

    model = _get_detection_model(exp_args.MODEL.N_CLASS,
                                 exp_args.MODEL.NAME,
                                 box_nms_thresh=args.nms,
                                 box_score_thresh=args.det)

    data_dir = 'data/'

    meta_dev = data_dir + 'dev.csv'
    labels_dev = pd.read_csv(meta_dev, na_filter=False)

    img_class_dict_dev = dict(zip(labels_dev.image_name,
                                  labels_dev.annotation))

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([transforms.Resize((600, 600)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=img_mean,
                                                               std=img_std)]
                                         )

    dataset_dev = ForeignObjectDataset(datafolder=data_dir+'dev/',
                                       datatype='dev',
                                       transform=data_transforms,
                                       labels_dict=img_class_dict_dev)
    data_loader_val = DataLoader(dataset_dev,
                                 batch_size=1,
                                 shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)

    device = torch.device('cuda:0')

    if args.weight == None:
        state_dict = os.path.join(exp_args.MODEL.SAVE_TO,
                              exp_args.MODEL.NAME,
                              exp_args.NAME + ".pt")
    else:
        state_dict = args.weight
    
    model.load_state_dict(torch.load(state_dict))
    model.to(device)

    model.eval()

    preds = []
    labels = []
    locs = []

    for image, label, width, height in tqdm(data_loader_val):
        
        image = list(img.to(device) for img in image)
        labels.append(label[-1])
        
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
            
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                center_x = (new_box[0] + new_box[2])/2
                center_y = (new_box[1] + new_box[3])/2
                center_points.append([center_x/600 * width[-1],center_y/600 * height[-1]])
            center_points_preds += new_scores.tolist()
            
            line = ''
            for i in range(len(new_boxes)):
                if i == len(new_boxes)-1:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1])
                else:
                    line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1]) +';'
            locs.append(line)

    cls_res = pd.DataFrame({'image_name': dataset_dev.image_files_list,
                            'prediction': preds})
    cls_res_path = os.path.join(exp_args.MODEL.SAVE_TO,
                                exp_args.MODEL.NAME,
                                "classification.csv")
    cls_res.to_csv(cls_res_path, columns=['image_name', 'prediction'],
                   sep=',', index=None)
    #print('classification.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset_dev.image_files_list,
                            'prediction': locs})
    loc_res_path = os.path.join(exp_args.MODEL.SAVE_TO,
                                exp_args.MODEL.NAME,
                                "localization.csv")
    loc_res.to_csv(loc_res_path, columns=['image_name', 'prediction'],
                   sep=',', index=None)
    #print('localization.csv generated.')

    pred = cls_res.prediction.values
    gt = labels_dev.annotation.astype(bool).astype(float).values

    acc = ((pred >= .5) == gt).mean()
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(subplot_kw=dict(xlim=[0, 1], ylim=[0, 1],
                                           aspect='equal'),
                           figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'ACC: {acc:.03}\nAUC: {roc_auc:.03}')
    _ = ax.legend(loc="lower right")
    _ = ax.set_title('ROC curve')
    ax.grid(linestyle='dashed')
    roc_curve = os.path.join(exp_args.MODEL.SAVE_TO,
                             exp_args.MODEL.NAME,
                             "roc_curve.eps")
    #plt.savefig(roc_curve)
    print("ACC: {} AUC: {}".format(acc, roc_auc))
