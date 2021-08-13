import sys
from flask import Flask, request, jsonify, render_template
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import VQADataset2, get_vqa_classes
from generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')