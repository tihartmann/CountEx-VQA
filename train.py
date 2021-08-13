import torch
from utils import save_checkpoint, load_checkpoint
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import config
import json
from dataset import VQADataset2, get_vqa_classes
from model.generator import Generator
from model.discriminator import Pix2PixDiscriminator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import random

sys.path.append("./VQA/vqa_pytorch/vqa/")
sys.path.append("./VQA/vqa_pytorch/")
sys.path.append('./VQA/vqa_pytorch/vqa/external/skip-thoughts.torch/pytorch/')
sys.path.append('./VQA/vqa_pytorch/vqa/external/pretrained-models.pytorch/')

from vqa_inference import MutanAttInference2

torch.backends.cudnn.benchmark = True
#set seed for reproduceability
torch.manual_seed(222)

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
    
def train_fn(disc, gen, vqa_model, loader, dataset, opt_disc, opt_gen, l2, cross_entropy, bce, g_scaler, d_scaler, epoch, L2_LAMBDA, CE_LAMBDA, tb):
    """
    Training function.

    Parameters
    ----------
        disc : Discriminator
            The discriminator of CountEx-VQA.
        gen : Generator
            The generator of CountEx-VQA.
        vqa_model : MutanAttInference2
            The pre-trained MUTAN model.
        loader : DataLoader
            PyTorch data loader.
        dataset : VQADataset2
            Instance of the VQA dataset.
        opt_disc : torch.optim
            Optimizer for the discriminator.
        opt_gen : torch.optim
            Optimizer for the generator.
        l2 : torch.nn.MSE() 
            Instance of L2 loss.
        cross_entropy : torch.nn.CrossEntropyLoss
            Instance of cross-entropy loss
        bce : torch.nn.BCEWithLogitsLoss
            Instance of binary cross-entropy loss
        g_scaler : torch.cuda.amp.GradScaler
            Scaler for the Generator for faster training and improved memory efficiency
        d_scaler : torch.cuda.amp.GradScaler
            Scaler for the Discriminator for faster training and improved memory efficiency.
        epoch : integer
            Number of current training epoch.
        L2_LAMBDA : float
            Weighting term for the L2-loss.
        CE_LAMBDA : float
            Weighting term for the cross-entropy loss.
        tb : torch.utils.tensorboard.SummaryWriter
            Summary writer to monitor training using Tensorboard.
    """
    
    print("epoch: ", epoch)
    
    # keep track of the losses
    G_losses = []
    D_losses = []
    D_fake_losses = []
    D_real_losses = []
    
    total_G_loss = 0
    total_D_loss = 0
    total_D_real = 0
    total_D_fake = 0
    total_L2 = 0 
    total_CE = 0
    iterations = 1
    
    #start the training loop for the given epoch
    loop = tqdm(loader, leave=True)
    for idx, (img, qid) in enumerate(loop):
        img = img.to(config.DEVICE)
        question = dataset.questions[qid.item()]
        
        # get attention and logits    
        with torch.no_grad():
            orig_im = img[0].clone()
            orig_im = trans_to_pil(orig_im)
       
        a1, orig_logits, q1, activations = vqa_model.infer(orig_im, question["question"])
        
        # select only needed classes (colors and shapes)        
        if not torch.argmax(orig_logits).item() in vqa_model.classes:
            print("class {} not in vqa_model.classes".format(torch.argmax(orig_logits).item()))
        orig_logits_new = orig_logits.clone()
        orig_logits_new = torch.Tensor(orig_logits_new.cpu().detach().numpy()[:,vqa_model.classes]).to(config.DEVICE)

        # normalize logits to be between [-1,1]
        orig_logits_new = norm_tensor(orig_logits_new)
        q1 = q1.clone()
        q1 = norm_tensor(q1)
        
        orig_logits, q1 = orig_logits.to(config.DEVICE), q1.to(config.DEVICE)
        answer = torch.tensor([torch.argmax(orig_logits_new).item()],device=config.DEVICE)

        # compute attention map, foreground object and background
        att, fg_real, bg_real = vqa_model.grad_cam(img[0].cpu(), orig_logits, activations)
        att = att.permute(2,0,1).unsqueeze(0).to(config.DEVICE)
        mask = att.clone()
        mask = normalize_mask(mask)
        #mask[att > 0.15] = 0.9
        #mask[att <= 0.2] = 0.2

        # get x_co and the background image
        fg_real = normalize_img(fg_real)
        bg_real = normalize_img(bg_real)
        fg_real, bg_real = fg_real.to(config.DEVICE), bg_real.to(config.DEVICE)
        
        #generate fake image
        with torch.cuda.amp.autocast():                    
            norm_img = normalize_img(img)
            y_fake = gen(torch.cat([fg_real, mask], 1), q1, orig_logits_new)
            
            generated = normalize_img((att * (y_fake * 0.5 + 0.5)) + (1.-att) * img)
            

        # Train Discriminator
        gen_im = generated[0].clone()
        gen_im = gen_im * 0.5 + 0.5
        gen_im = trans_to_pil(gen_im)

        # get VQA output for the generated image
        a2, pred_logits, q2, activations2 = vqa_model.infer(gen_im, question["question"])
        pred_logits_new = pred_logits.clone()
        pred_logits_new = torch.Tensor(pred_logits.cpu().detach().numpy()[:,vqa_model.classes]).to(config.DEVICE)
        
        #normalize logits to be between [-1,1]
        pred_logits_new = norm_tensor(pred_logits_new)
        
        # compute cross-entropy loss
        CROSS_ENTROPY_LOSS = -torch.sigmoid(cross_entropy(pred_logits_new, answer)) * CE_LAMBDA # negated cross entropy
        
        if epoch > 0 and idx % 1 == 0:
            # discriminator loss
            with torch.cuda.amp.autocast():
                if epoch >= 0:
                    D_real = disc(norm_img, orig_logits_new)
                    D_fake = disc(generated.detach(), pred_logits_new) #detach to avoid breaking the computational graph when using optimizer.step() on the discriminator
                else:
                    norm_img = normalize_img(img)
                    D_real = disc(fg_real, orig_logits_new)
                    D_fake = disc(y_fake.detach(), pred_logits_new)


                # add noise because the discriminator learns too quickly
                if random.uniform(0,1) < 0.0: # specify value greater than 0.0 to randomly flip the labels of the real image
                    D_real_loss = bce(D_real, torch.zeros_like(D_real))
                else:           
                    D_real_loss = bce(D_real, torch.ones_like(D_real)*0.99) # change to 0.9 to apply one-sided label smoothing
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

                D_loss = (D_real_loss + D_fake_loss) / 2
                
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            
            
        elif epoch == 0:
            D_real = torch.tensor(0)
            D_fake = torch.tensor(0)
            D_loss = torch.tensor(0)
        
        
        # Train Generator
        with torch.cuda.amp.autocast():

            if epoch == 0:
                L2 = l2(generated, normalize_img(img)) * 1
                G_loss = L2 # warmup LingUNet for 1 epoch
            else:
                D_fake = disc(generated, pred_logits_new)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake)) # add one sided label smoothing
                
                L2 =  l2(normalize_img((1.-att) * (generated * 0.5 + 0.5)), bg_real) * L2_LAMBDA # which lambda to choose??
               
                G_loss = G_fake_loss + CROSS_ENTROPY_LOSS + L2


        

        gen.zero_grad() # changed from opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
   
        if idx % 50 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_loss=G_loss.mean().item(),
                CROSS_ENTROPY_LOSS=CROSS_ENTROPY_LOSS.mean().item(),
            )
            G_losses.append(G_loss.mean().item())
            D_losses.append(torch.sigmoid(D_loss).mean().item())
            D_fake_losses.append(torch.sigmoid(D_fake).mean().item())
            D_real_losses.append(torch.sigmoid(D_real).mean().item())
        # update total loss
        total_G_loss = total_G_loss + G_loss.mean().item()
        total_D_loss = total_D_loss + torch.sigmoid(D_loss).mean().item()
        total_D_real = total_D_real + torch.sigmoid(D_real).mean().item()
        total_D_fake = total_D_fake + torch.sigmoid(D_fake).mean().item()
        total_L2 = total_L2 + L2.mean().item()
        total_CE = total_CE + CROSS_ENTROPY_LOSS.mean().item()
        iterations = iterations + 1

        if idx % 500 == 0:
            ans = json.loads(a1)["ans"][0]
            ans_new = json.loads(a2)["ans"][0]

            image_to_save = generated * 0.5 + 0.5
            try:
                save_image(image_to_save, f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}/y_gen_q_{str(question) + ' '+ans_new}_{epoch}.png")
                save_image(img, f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}/orig_q_{str(question)+' '+ans}_{epoch}.png")
            except:
                pass
            del ans,image_to_save
        #del orig_im, question, a1, orig_logits, q1, activations, orig_logits_new, att, fg_real, bg_real,gen_im, a2, pred_logits, q2, activations2, pred_logits_new, norm_img, answer, generated, qid, 
        #print(gc.get_objects(generation=2)) # garbage collector
    
    tb.add_scalar("Discriminator Loss", total_D_loss / iterations, epoch)
    tb.add_scalar("Generator Loss", total_G_loss / iterations, epoch)
    tb.add_scalar("D_Real Loss", total_D_real / iterations, epoch)
    tb.add_scalar("D_Fake Loss", total_D_fake / iterations, epoch)
    
    return G_losses, D_losses, D_fake_losses, D_real_losses
        
            
def main():   
    """Main function to start training procedure of CountEx-VQA."""
    
    # set up folders to save example results during training
    if not os.path.isdir("./evaluation"):
        os.mkdir("./evaluation")
        os.mkdir("./evaluation/training")

    disc = Pix2PixDiscriminator().to(config.DEVICE) 
    gen = Generator(features=64).to(config.DEVICE)
    
    # optimizer for the discriminator
    opt_disc = optim.Adam(disc.parameters(), lr=config.D_LEARNING_RATE, betas=(0.5, 0.999),) #betas and LR based on pix2pix paper
    opt_gen = optim.Adam(gen.parameters(), lr=config.G_LEARNING_RATE, betas=(0.5, 0.999))#, betas=(0.5, 0.999),) # betas and LR based on pix2 pix paper
    L2_LOSS = nn.MSELoss()
    CROSS_ENTROPY = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    #vqa_model = None
   
    train_dataset = VQADataset2(root_dir="./", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    
     #with torch.no_grad():
    vqa_model = MutanAttInference2(dir_logs='./VQA/vqa_pytorch/logs/vqa/mutan_att_trainval', config='./VQA/vqa_pytorch/options/vqa/mutan_att_trainval.yaml')
    classes = get_vqa_classes(train_dataset, vqa_model)
    vqa_model.classes = classes
    vqa_model.model.to(config.DEVICE)
    vqa_model.model.eval() #THIS IS IMPORTANT!
    
    g_scaler = torch.cuda.amp.GradScaler() #faster and uses less VRAM, same results
    d_scaler = torch.cuda.amp.GradScaler()
    
    val_dataset = VQADataset2(root_dir="./", mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    all_losses = {"G_losses": [], "D_losses": [], "D_fake": [], "D_real": []}
   
    for CE_LAMBDA in config.CE_LAMBDA:
        for L2_LAMBDA in config.L2_LAMBDA:
             # tensorboard
            tb = SummaryWriter(log_dir=f'runs/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}_{str(datetime.now())}')
            if not os.path.isdir(f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}"):
                os.mkdir(f"evaluation/training/L2-lambda_{L2_LAMBDA}_CE-lambda_{CE_LAMBDA}")
            if config.LOAD_MODEL:
                load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
                load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
            
            loss_file = f"evaluation/losses_{str(datetime.now())}_l2lambda_{L2_LAMBDA}_CE_{CE_LAMBDA}_GLR_{config.G_LEARNING_RATE}_DLR_{config.D_LEARNING_RATE}.json"
     
            for epoch in range(config.NUM_EPOCHS):
                G_losses, D_losses, D_fake_losses, D_real_losses = train_fn(disc, gen, vqa_model, train_loader, train_dataset, opt_disc, opt_gen, L2_LOSS, CROSS_ENTROPY, BCE, g_scaler, d_scaler, epoch, L2_LAMBDA, CE_LAMBDA, tb)
                all_losses["G_losses"].append(G_losses)
                all_losses["D_losses"].append(D_losses)
                all_losses["D_fake"].append(D_fake_losses)
                all_losses["D_real"].append(D_real_losses)
                if config.SAVE_MODEL and epoch % 5 == 0:
                    save_checkpoint(gen, opt_gen, filename=f'gen_L1_{L2_LAMBDA}_CE_{CE_LAMBDA}_epoch_{epoch}.pth.tar')
                    save_checkpoint(disc, opt_disc, filename=f'disc_{L2_LAMBDA}_CE_{CE_LAMBDA}_epoch_{epoch}.pth.tar')

                #save_some_examples(gen, val_loader, epoch, folder="evaluation")
                

                with open(loss_file, 'w') as f:
                    json.dump(all_losses, f)
            tb.close()

if __name__ == "__main__":
    main()