import numpy as np
import config
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import json
import sys
import torch
import spacy
import cv2
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
        
class VQADataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        annotations_path = os.path.join("data/annotations/questions/", mode)
        self.annotations_dir = os.path.join(self.root_dir, annotations_path)
        self.nlp = spacy.load('en_core_web_lg')
        
        vg_classes_path = os.path.join(self.root_dir, 'py-bottom-up-attention/demo/data/genome/1600-400-20')
        self.vg_classes = []
        with open(os.path.join(vg_classes_path, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_classes.append(object.split(',')[0].lower().strip())
                
        self.vg_attrs = []
        with open(os.path.join(vg_classes_path, 'attributes_vocab.txt')) as f:
            for object in f.readlines():
                self.vg_attrs.append(object.split(',')[0].lower().strip())
        
        # questions        
        if mode == "train":
            self.questions_path = os.path.join(self.root_dir,'data/vqa/OpenEnded_mscoco_train2014_questions.json')
        elif mode == "val":
            self.questions_path = os.path.join(self.root_dir, 'data/vqa/OpenEnded_mscoco_val2014_questions.json')
        self.questions = json.load(open(self.questions_path))["questions"]
        self.questions = [q for q in self.questions if q["question"].lower().startswith("what color")]
        print(len(self.questions))
        if mode == "train":
            random.Random(4).shuffle(self.questions)
            self.questions = self.questions
                                                        
    def __len__(self):
        return len(self.questions)
    
      
    def __getitem__(self, index):
        question_item = self.questions[index]
        annotation_file = str(question_item["question_id"]) + ".npz"
        annotation_path = os.path.join(self.annotations_dir, annotation_file)
        
        # load data
        with np.load(annotation_path, allow_pickle=True) as data:
            img = cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB)
            #img = data["img"]
            info = dict(data["info"].tolist())
            logits = torch.tensor(data["logits"][0])
            q_emb = torch.tensor(data["q_emb"][0])
            #print(info["img_path"])
        
        # select question critical objects
        keep_boxes = {}
        doc = self.nlp(question_item["question"])
        for i in range(len(info["bboxes"])):
            obj = self.vg_classes[info["pred_classes"][i]]
            attr = self.vg_attrs[info["attr_classes"][i]]
            attr_obj = attr + " " + obj
            
            # similarity
            for nc in doc.noun_chunks:
                sim = self.nlp(attr_obj).similarity(nc)
                if i in keep_boxes.keys():
                    if sim > keep_boxes[i]:
                        keep_boxes[i] = sim
                else:
                    keep_boxes[i] = sim
                #if sim > 0.6 and i not in keep_boxes:
                    #keep_boxes.append(i)
        keep_boxes = dict(sorted(keep_boxes.items(), key=lambda item: item[1], reverse=True)[:3])
        
        # generate mask
        mask = np.ones((img.shape[0], img.shape[1], 1), np.uint8)
        applied_mask = False
        for k,v in keep_boxes.items():
            if v > 0.6:
                mask = npmask(info["bboxes"][k], img.shape[0], img.shape[1], mask)
                applied_mask = True
        if not applied_mask:
            mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
        
        #orig_image = config.both_transform(image=img)["image"]
        masked_image = img * mask #cv2.bitwise_and(img,mask)
        #plt.imshow(masked_image)
        
        orig_image = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
        masked_image = cv2.resize(masked_image, (256,256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        mask = gray == 0
        
        
        #mask = config.both_transform(image=mask)["image"]
        
        #orig_image = torch.Tensor(orig_image)
        #masked_image = torch.Tensor(masked_image)
        with torch.no_grad():
            tran = transforms.ToTensor()
            orig_image = tran(orig_image)
        #masked_image = tran(masked_image)
        #print(mask)
            mask = torch.ByteTensor(mask)
            q_id = torch.tensor([index])

        return orig_image, mask, logits, q_emb, q_id
                              