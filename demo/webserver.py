from flask import Flask, request, jsonify, render_template, flash, send_file
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import VQADataset2, get_vqa_classes
from model.generator import Generator
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import json
from PIL import Image
from io import BytesIO
import base64
import secrets

from VQA.vqa_pytorch.vqa_inference import MutanAttInference2
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

secret_key = secrets.token_hex(16)


app = Flask(__name__)
app.secret_key = secret_key



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

# functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    return img, att, generated * 0.5 + 0.5, a1, a2, question

def get_attention(image, normalized_heat_map):
    buff = BytesIO()
    plt.figure(figsize=(2.56,2.56))
    plt.imshow(image)
    plt.imshow(255* normalized_heat_map, alpha=0.6, cmap="turbo")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buff, format="JPEG", dpi=100, bbox_inches="tight", pad_inches=0)
    buff.seek(0)
    heat_img_base64 = base64.b64encode(buff.read()).decode('ascii')
    buff.close()
    return heat_img_base64

@app.route('/')
def home():
    return render_template('index.html')

def serve_pil_image(pil_image):
    pass

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            print(request.files)
            if 'inputImage' not in request.files:
                print("image not there!")
                flash('Please upload an image!')
                return render_template('index.html')
            if request.form.get('inputQuestion') == "":
                flash('Please enter a question!')
                return render_template('index.html')
            
            # get form data
            img = request.files['inputImage']
            if img.filename == '':
                flash('Please upload a valid image!')
                return render_template('index.html')
            if not allowed_file(img.filename):
                flash('Not a valid image file!')
                return render_template('index.html')
            if img and allowed_file(img.filename):
                img = img.read()
                img = base64.b64encode(img).decode('ascii')
                visual_Tensor = trans_to_Tensor(Image.open(BytesIO(base64.b64decode(img))))
                
                question = request.form.get('inputQuestion')
                
                # make inference
                img_tensor, att, counterfactual, a1, a2, _ = infer(resize(visual_Tensor), question, dataset=train_dataset)
                orig_ans = json.loads(a1)["ans"][0]
                counter_ans = json.loads(a2)["ans"][0]
                # get heatmap
                img_tensor = img_tensor[0].permute(1,2,0).detach().cpu().numpy()
                heat_map = get_attention(np.array(img_tensor), att[0].permute(1,2,0).detach().cpu())
                # convert counterfactual to base64

                counterfactual = trans_to_pil(counterfactual[0])
                buff = BytesIO()
                counterfactual.save(buff, format="JPEG")
                counterfactual = base64.b64encode(buff.getvalue()).decode('ascii')
                buff.close()
                return render_template(
                    'predict.html', 
                    question=question, 
                    original_image=f'<img src="data:image/jpg;base64,{img}" class="img-fluid" width="256" height="256"/>', 
                    orig_ans=orig_ans,
                    heat_map=f'<img src="data:image/jpg;base64,{heat_map}" class="img-fluid" width="256" height="256"/>',
                    counterfactual=f'<img src="data:image/jpg;base64,{counterfactual}" class="img-fluid" width="256" height="256"/>',
                    counter_ans=counter_ans,
                )
    except:

        flash('Something went wrong!')
        return render_template('index.html')

if __name__ == "__main__":
    app.run()