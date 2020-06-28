

## Test Local

```
python src/submit.py image_path.csv predictions_classification.csv predictions_localization.csv

```

## Codalab flow

1. Get data

```
cl add bundle object-CXR-utlis//valid_image .
cl add bundle object-CXR-utlis//image_path.csv .
```

2. Upload folder

3. Run 

```
cl run image_path.csv:image_path.csv valid_image:valid_image src:src "python3 src/submit.py image_path.csv predictions_classification.csv predictions_localization.csv"  
 -n run-predictions

```

4. Extract Predictions

```
cl make run-predictions/predictions_classification.csv  -n predictions-classification-{MODELNAME}
cl make run-predictions/predictions_localization.csv -n predictions-localization-{MODELNAME}
```

5. Validate Predictions

```
cl add bundle object-CXR-utlis//valid_gt.csv .
cl add bundle object-CXR-utlis//program .
cl run valid_gt.csv:valid_gt.csv run-predictions:run-predictions program:program "python3 program/evaluate_auc.py run-predictions/predictions_classification.csv valid_gt.csv score.txt" -n score_auc --request-docker-image yww211/codalab:foreginobjv2
cl run valid_gt.csv:valid_gt.csv run-predictions:run-predictions program:program "python3 program/evaluate_froc.py run-predictions/predictions_localization.csv valid_gt.csv score.txt" -n score_froc --request-docker-image yww211/codalab:foreginobjv2
```
