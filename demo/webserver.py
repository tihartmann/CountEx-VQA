from flask import Flask, request, jsonify, render_template, flash, send_file
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import VQADataset2, get_vqa_classes
from model.generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import re
from PIL import Image
from io import BytesIO
import base64
from VQA.vqa_pytorch.vqa_inference import MutanAttInference2
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

resize = transforms.Compose([
            transforms.Resize((256,256)),
            #transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
        )
normalize_img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
normalize_mask = transforms.Normalize((0.5),(0.5))
trans_to_pil = transforms.ToPILImage()
trans_to_Tensor = transforms.ToTensor()
device = 'cuda'

def norm_tensor(x):
    x = x[0]
    if not all(x == 0):
        x -= x.min()
        x /= x.max()
        x = 2*x - 1
        return(x[None,:])
    else:
        return x

def infer(img, question, dataset):
    img = img[None,:,:,:]
    img = resize(img)
    img = img.to(device)
    print(img.shape)
    #question = dataset.questions[qid.item()]
    # get attention and logits    
    with torch.no_grad():
        orig_im = img[0].clone()
        #plt.imshow(orig_im.cpu().permute(1,2,0))
        #plt.savefig("realreal.jpg")
        #orig_im = orig_im.permute(2,0,1)
        orig_im = trans_to_pil(orig_im)

    a1, orig_logits, q1, activations = vqa_model.infer(orig_im, question)
    
    # select only needed classes (colors and shapes)   
     # select only needed classes (colors and shapes)        
    if not torch.argmax(orig_logits).item() in vqa_model.classes:
        print("class {} not in vqa_model.classes".format(torch.argmax(orig_logits).item()))
    orig_logits_new = orig_logits.clone()
    orig_logits_new = torch.Tensor(orig_logits_new.cpu().detach().numpy()[:,vqa_model.classes]).to(device)
    # normalize logits to be between [-1,1]
    orig_logits_new = norm_tensor(orig_logits_new)
    orig_logits_new, q1 = orig_logits_new.to(device), q1.to(device)
    answer = torch.tensor([torch.argmax(orig_logits_new).item()],device=device)
    # compute attention map, foreground object and background
    att, fg_real, bg_real = vqa_model.grad_cam(img[0].cpu(), orig_logits, activations)
    att = att.permute(2,0,1).unsqueeze(0).to(device)
    mask = att.clone()
    mask = normalize_mask(mask)
    
    # get x_co and the background image
    fg_real = normalize_img(fg_real)


    bg_real = normalize_img(bg_real)
   
    fg_real, bg_real = fg_real.to(device), bg_real.to(device)
    with torch.cuda.amp.autocast():                    
        norm_img = normalize_img(img)
        y_fake = gen(torch.cat([norm_img, mask], 1), q1, orig_logits_new)
        generated = normalize_img((att * (y_fake * 0.5 + 0.5)) + (1.-att) * img)
    
    gen_im = generated[0].clone()
    gen_im = gen_im * 0.5 + 0.5
    gen_im = trans_to_pil(gen_im)

    a2, pred_logits, q2, activations2 = vqa_model.infer(gen_im, question)
    #pred_logits_new = pred_logits.clone()
    #pred_logits_new = torch.Tensor(pred_logits.cpu().detach().numpy()[:,vqa_model.classes]).to(device)
    #normalize logits to be between [-1,1]
    #pred_logits_new = norm_tensor(pred_logits_new)
    return img, generated * 0.5 + 0.5, a1, a2, question

app = Flask(__name__)


# laod generator
gen = Generator().to(device)
checkpoint = torch.load('../model/generator.pth.tar')
gen.load_state_dict(checkpoint["state_dict"])
gen.eval()

# load VQA model
vqa_model = MutanAttInference2(dir_logs='../VQA/vqa_pytorch/logs/vqa/mutan_att_trainval', config='../VQA/vqa_pytorch/options/vqa/mutan_att_trainval.yaml')
train_dataset = VQADataset2(root_dir="../", mode="train")

classes = get_vqa_classes(train_dataset, vqa_model)
vqa_model.classes = classes
vqa_model.model = vqa_model.model.to(device)
vqa_model.model.eval()


@app.route('/')
def home():
    return render_template('index.html')

def serve_pil_image(pil_image):
    pass

@app.route('/predict', methods=['GET','POST'])
def predict():
        
    # get form data
    img_file = request.files['inputImage'].filename
    img = request.files['inputImage'].read()
    img = base64.b64encode(img).decode('ascii')
    visual_Tensor = trans_to_Tensor(Image.open(BytesIO(base64.b64decode(img))))
    
    question = request.form.get('inputQuestion')
    
    # make inference
    _, counterfactual, a1, a2, _ = infer(visual_Tensor, question, dataset=train_dataset)

    # convert counterfactual to base64
    counterfactual = counterfactual[0].cpu().numpy()
    counterfactual = base64.b64encode(counterfactual)
    print(a1)
    return render_template(
        'index.html', 
        question=question, 
        original_image=f'<img src="data:image/jpg;base64,{img}" class="img-fluid" width="256" height="256"/>', 
        counterfactual=f'<img src="data:image/jpg;base64,{counterfactual}" class="img-fluid" width="256" height="256"/>'    
    )

if __name__ == "__main__":
    app.run(debug=True)