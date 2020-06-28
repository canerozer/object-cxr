from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def fasterrcnn_resnet_fpn(pretrained=False, progress=True, resnet='resnet50',
                          num_classes=91, pretrained_backbone=False, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(resnet, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    # if pretrained:
    #     target_url = model_urls['fasterrcnn_' + resnet + '_fpn_coco']
    #     state_dict = load_state_dict_from_url(target_url, progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def _get_detection_model(num_classes, resnet, **kwargs):
    model = fasterrcnn_resnet_fpn(pretrained=False, resnet=resnet, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
