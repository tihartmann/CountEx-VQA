## Script for downloading data and models

# VQAv1 Input Questions
mkdir ./data/
wget -P ./data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
wget -P ./data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip

# COCO images
mkdir ./data/coco
wget -P ./data/coco http://images.cocodataset.org/zips/train2014.zip
wget -P ./data/coco http://images.cocodataset.org/zips/val2014.zip