from flask import Flask, request, jsonify, render_template
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import VQADataset2, get_vqa_classes
from model.generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

from VQA.vqa_pytorch.vqa_inference import MutanAttInference2

app = Flask(__name__)

# laod generator
gen = Generator()
checkpoint = torch.load('../model/generator.pth.tar')
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

# load VQA model
vqa_model = MutanAttInference2(dir_logs='../VQA/vqa_pytorch/logs/vqa/mutan_att_trainval', config='../VQA/vqa_pytorch/options/vqa/mutan_att_trainval.yaml')
train_dataset = VQADataset2(root_dir="../", mode="train")

classes = get_vqa_classes(train_dataset, vqa_model)
vqa_model.classes = classes
vqa_model.model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    img = features[0]
    question = features[1]
    infer(img,question)


normalize_img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
normalize_mask = transforms.Normalize((0.5),(0.5))
trans_to_pil = transforms.ToPILImage()

def norm_tensor(x):
    x = x[0]
    if not all(x == 0):
        x -= x.min()
        x /= x.max()
        x = 2*x - 1
        return(x[None,:])
    else:
        return x

def infer(img, q, dataset=train_dataset):
    print(img)
    print(q)
