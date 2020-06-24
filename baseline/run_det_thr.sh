for i in $(seq -f "0.%02g" 5 5 95);
do
    for j in $(seq -f "0.%02g" 5 5 95);
    do
        echo "Detection threshold: $i NMS threshold: $j" | tee -a records;
        python3 eval_fasterrcnn.py --yaml=configs/faster_rcnn_R_101_FPN_1x.yaml --nms=$i --det=$j | tee -a records;
        python3 froc.py data/dev.csv models/resnet101/localization.csv | tee -a records 

    done
done
