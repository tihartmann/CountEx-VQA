## Script for downloading coco data
mkdir ./data

# VQAv1 questions and annotations
mkdir ./data/vqa
wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
wget -P ./data/vqa https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
unzip -d ./data/vqa/ ./data/vqa/Questions_Train_mscoco.zip
unzip -d ./data/vqa/ ./data/vqa/Questions_Val_mscoco.zip

wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
wget -P ./data/vqa/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip
unzip -d ./data/vqa/ ./data/vqa/Annotations_Train_mscoco.zip
unzip -d ./data/vqa/ ./data/vqa/Annotations_Val_mscoco.zip

# COCO images
mkdir ./data/coco
wget -P ./data/coco http://images.cocodataset.org/zips/train2014.zip
wget -P ./data/coco http://images.cocodataset.org/zips/val2014.zip
unzip -d ./data/coco/ ./data/coco/train2014.zip
unzip -d /.data/coco/ ./data/coco/val2014.zip

# pretrained MUTAN model
mkdir -p ./VQA/vqa_pytorch/logs/
mkdir -p ./VQA/vqa_pytorch/logs/vqa
wget http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mutan_att_trainval.zip 
unzip -d ./VQA/vqa_pytorch/logs/vqa/mutan_att_trainval ./VQA/vqa_pytorch/logs/vqa/mutan_att_trainval.zip 