# CUB-200-2010 Classification

* [competition url](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07)


# Configuration of experiment
file `params.json` under `./experiments/baseline/resnet50/` folder can tune the configuration
# prerequisite 
1. download dataset
dataset structure
* 2021VRDL_HW1_datasets/
    * testing_images/*.jpg
    * training_images/*.jpg
    classes.txt
    testing_img_order.txt
    training_labels.txt 



# Training
python3 main.py --model_dir ./experiments/baseline/resnet50/ --dataset {the path to 2021VRDL_HW1_datasets}
* ex: python3 main.py  --dataset ../2021VRDL_HW1_datasets

# Evaluation
python3 inference.py --dataset --dataset {the path to 2021VRDL_HW1_datasets}



# Reproducing Submission

To reproduct my submission without training, do the following steps:

1. download  2021VRDL_HW1_datasets
1. trained model 
2. python3 inference.py  --dataset {the path to 2021VRDL_HW1_datasets}
3. answer.txt will be generated in the current folder


### detail
1. training code:
code: main.py train.py utils.py cub2010.py loss.py models/basemodel.py 
config:hw1/code/CUB200-2010-Classification/experiments/baseline/resnet50/params.json
2. evaluation code:
code: inference.py utils.py cub2010.py loss.py  models/basemodel.py 
config:hw1/code/CUB200-2010-Classification/experiments/baseline/resnet50/params.json
3. pretrained model:resnet50

# Result
My model achieves the following performance on :

| Model   | Top-1 Accuracy |
| -------- | -------- |
| ResNet50 | 0.719749 |

