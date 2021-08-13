import numpy as np
import os
from torch.utils.data import Dataset
import json
import torch
from torchvision import transforms
import random
from PIL import Image

def npmask(bbox, height, width, mask=np.nan):
    """if mask == np.nan:
        mask = np.zeros((height, width, 1), np.int)"""
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1 
    mask[int(bbox[1]):int(bbox[3]),
         int(bbox[0]):int(bbox[2]),:] = 0.
    return mask


def q2img_path(coco_path, q, split="train"):
    img_id = q['image_id']
    id_len = len(str(img_id))
    img_path = ""
    if split == "val":
        file_name = "val2014/COCO_val2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    elif split == "train":
        file_name = "train2014/COCO_train2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    return img_path

def q2img_path(coco_path, q, split="val"):
    img_id = q['image_id']
    id_len = len(str(img_id))
    img_path = ""
    if split == "val":
        file_name = "val2014/COCO_val2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    elif split == "train":
        file_name = "train2014/COCO_train2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    return img_path

def npmask(bbox, height, width, mask=np.nan):
    """if mask == np.nan:
        mask = np.zeros((height, width, 1), np.int)"""
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1 
    mask[int(bbox[1]):int(bbox[3]),
         int(bbox[0]):int(bbox[2]),:] = 0.
    return mask


def q2img_path(coco_path, q, split="train"):
    img_id = q['image_id']
    id_len = len(str(img_id))
    img_path = ""
    if split == "val":
        file_name = "val2014/COCO_val2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    elif split == "train":
        file_name = "train2014/COCO_train2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    return img_path

def q2img_path(coco_path, q, split="val"):
    img_id = q['image_id']
    id_len = len(str(img_id))
    img_path = ""
    if split == "val":
        file_name = "val2014/COCO_val2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    elif split == "train":
        file_name = "train2014/COCO_train2014_{}{}.jpg".format((12-id_len) * str(0), str(img_id))
        img_path = os.path.join(coco_path, file_name)
    return img_path

class VQADataset2(Dataset):
    """
    Class to initialize a VQA dataset.

    Parameters
    ----------
        root_dir : string
            Root directory where the data is stored.
        mode : string, optional
            Specifies whether to use the training or validation split of the VQA dataset.
            Has to be in ["train","val"].
            The default value is "train".
        numbers_only : boolean, optional
            If True, only number-based questions are loaded from the dataset.
            If False, color and shape-based questions are loaded as described in the Thesis.
            The default value is False.
    """
    def __init__(self, root_dir, mode="train", numbers_only=False):
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
        )
        # questions        
        if mode == "train":
            self.questions_path = os.path.join(self.root_dir,'data/vqa/OpenEnded_mscoco_train2014_questions.json')
            self.annotations_path = os.path.join(self.root_dir, 'data/vqa/mscoco_train2014_annotations.json')
        elif mode == "val":
            self.questions_path = os.path.join(self.root_dir, 'data/vqa/OpenEnded_mscoco_val2014_questions.json')
            self.annotations_path = os.path.join(self.root_dir, 'data/vqa/mscoco_val2014_annotations.json')

        with open(self.questions_path) as question_file:
            self.questions = json.load(question_file)["questions"]
        with open(self.annotations_path) as annotations_file:
            self.annotations = json.load(annotations_file)["annotations"]
        
        
        if numbers_only:
            self.questions = [q for q in self.questions if q["question"].lower().startswith("what number")]
        else:
            self.questions = [q for q in self.questions if q["question"].lower().startswith("what color") or q["question"].lower().startswith("what shape")]
        print(len(self.questions))
        
        
        if mode == "train":
            random.Random(4).shuffle(self.questions)
            self.questions = self.questions
        self.createIndex()
        
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question_item = self.questions[index]
        
        img_path = q2img_path(os.path.join(self.root_dir, 'data/coco/'), question_item, split=self.mode)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        #img = img#.permute(1,2,0)
        q_id = torch.tensor([index])
        
        return img, q_id   

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.annotations}
        qa =  {ann['question_id']: [] for ann in self.annotations}
        qqa = {ann['question_id']: [] for ann in self.annotations}
        for ann in self.annotations:
            imgToQA[ann['image_id']] += [ann]
            qa[ann['question_id']] = ann
        for ques in self.questions:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

def get_vqa_classes(dataset, vqa_model):
    classes = []
    for q in dataset.questions:
        qid = q["question_id"]
        ans = dataset.qa[qid]["answers"]
        for a in ans:
            try:
                aid = vqa_model.trainset.ans_to_aid[a["answer"]]
                if aid not in classes:
                    classes.append(aid)
            except:
                pass
    print(f"Number of VQA classes: {len(classes)}")
    return classes