## Script for downloading coco data
mkdir ./data

# VQAv1 questions and annotations
mkdir ./data/vqa
wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
wget -P ./data/vqa https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Train_mscoco.zip
unzip Questions_Val_mscoco

wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip
unzip Annotations_Train_mscoco
unzip Annotations_Val_mscoco

# COCO images
mkdir ./data/coco
wget -P ./data/coco http://images.cocodataset.org/zips/train2014.zip
wget -P ./data/coco http://images.cocodataset.org/zips/val2014.zip
unzip ./data/coco/train2014.zip
unzip ./data/coco/val2014.zip