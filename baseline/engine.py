import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, 
                    print_freq, my_logger=None, name=None, env_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    warmup_lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                        warmup_iters,
                                                        warmup_factor)

    for images, targets, name in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            for img, t in zip(images, targets):
                print(img.shape)
                print(t)
                print(name)
            sys.exit(1)

        if my_logger:
            my_logger.scalar(loss_value, env=env_name, win="Loss",
                             trace=name, xlabel="Iteration")

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_lr_scheduler:
            warmup_lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
