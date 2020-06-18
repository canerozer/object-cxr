python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_18_FPN_1x.yaml 
python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_34_FPN_1x.yaml 
python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_50_FPN_1x.yaml 
python3 train_fasterrcnn.py --yaml=configs/faster_rcnn_R_101_FPN_1x.yaml

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_18_FPN_1x.yaml
python3 froc.py data/dev.csv localization.csv

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_34_FPN_1x.yaml
python3 froc.py data/dev.csv localization.csv

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_50_FPN_1x.yaml 
python3 froc.py data/dev.csv localization.csv

python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_101_FPN_1x.yaml
python3 froc.py data/dev.csv localization.csv

