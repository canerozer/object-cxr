#python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_18_FPN_2x_DA.yaml 
#python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_34_FPN_2x_DA.yaml 
#python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_50_FPN_2x_DA.yaml 
python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_101_FPN_2x_DA.yaml
python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_152_FPN_2x_DA.yaml

#python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_18_FPN_2x_DA.yaml
#python3 froc.py data/dev.csv localization.csv

#python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_34_FPN_2x_DA.yaml
#python3 froc.py data/dev.csv models/resnet34/localization.csv

#python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_50_FPN_2x_DA.yaml 
#python3 froc.py data/dev.csv models/resnet50/localization.csv

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_101_FPN_2x_DA.yaml
python3 froc.py data/dev.csv models/resnet101/localization.csv

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_152_FPN_2x_DA.yaml
python3 froc.py data/dev.csv models/resnet152/localization.csv
