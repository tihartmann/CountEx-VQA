import os
import time
import yaml
import json
import argparse
import re
import base64
import torch
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
from pprint import pprint
import numpy as np
from skimage import transform
from scipy import ndimage
from skimage import io

import torchvision.transforms as transforms
import vqa.lib.utils as utils
import vqa.datasets as datasets
import vqa.models as models
import vqa.models.convnets as convnets
from vqa.datasets.vqa_processed import tokenize_mcb

def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.pth.tar'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc1   = 0
        exp_logger  = None
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_acc1' in info:
            best_acc1 = info['best_acc1']
        else:
            print('Warning train.py: no best_acc1 to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    if optimizer is not None and os.path.isfile(path_ckpt_optim):
        optim_state = torch.load(path_ckpt_optim)
        optimizer.load_state_dict(optim_state)
    else:
        print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
              .format(path_ckpt, start_epoch, best_acc1))
    return start_epoch, best_acc1, exp_logger

def process_question(question_str, trainset):
    question_tokens = tokenize_mcb(question_str)
    question_data = torch.LongTensor(1, len(question_tokens))
    for i, word in enumerate(question_tokens):
        if word in trainset.word_to_wid:
            question_data[0][i] = trainset.word_to_wid[word]
        else:
            question_data[0][i] = trainset.word_to_wid['UNK']
    if torch.cuda.is_available():
        question_data = question_data.cuda(non_blocking=True)
    with torch.no_grad():
        question_input = Variable(question_data, volatile=True)
    #print('question', question_str, question_tokens, question_data)
    
    return question_input

def process_answer(answer_var, trainset, model):
    with torch.no_grad():
        answer_sm = torch.nn.functional.softmax(answer_var.data[0].cpu(), dim=0)
        max_, aid = answer_sm.topk(5, 0, True, True)
    ans = []
    val = []
    for i in range(5):
        ans.append(trainset.aid_to_ans[aid.data[i]])
        val.append(max_.data[i].cpu().item())
    """
    att = []
    for x_att in model.list_att:
        img = x_att.view(1,14,14).cpu()
        img = transforms.ToPILImage()(img)
        buffer_ = BytesIO()
        img.save(buffer_, format="PNG")
        img_str = base64.b64encode(buffer_.getvalue()).decode()
        img_str = 'data:image/png;base64,'+img_str
        att.append(img_str)
        buffer_.close()
    """
    answer = {'ans':ans,'val':val} # 'att': att
    answer_str = json.dumps(answer)

    return answer_str

def process_visual(visual_pil,transform, options, cnn):
    #visual_strb64 = re.sub('^data:image/.+;base64,', '', visual_strb64)
    #visual_PIL = Image.open(BytesIO(base64.b64decode(visual_strb64)))
    visual_PIL = visual_pil
    visual_tensor = transform(visual_PIL)
    visual_data = torch.FloatTensor(1, 3,
                                       visual_tensor.size(1),
                                       visual_tensor.size(2))
    visual_data[0][0] = visual_tensor[0]
    visual_data[0][1] = visual_tensor[1]
    visual_data[0][2] = visual_tensor[2]
    #visual_data = visual_pil
    #print('visual', visual_data.size(), visual_data.mean())
    if torch.cuda.is_available():
        visual_data = visual_data.cuda(non_blocking=True)
    with torch.no_grad():
        visual_input = Variable(visual_data, volatile=True)
        visual_features = cnn(visual_input)
    if 'NoAtt' in options['model']['arch']:
        nb_regions = visual_features.size(2) * visual_features.size(3)
        with torch.no_grad():
            visual_features = visual_features.sum(3).sum(2).div(nb_regions).view(-1, 2048)
    return visual_features


    
    
class MutanAttInference2():
    """
    MutanAtt model wrapper
    """

    def __init__(self, classes_idxs=None, dir_logs='logs/vqa/mutan_att_trainval', config='options/vqa/mutan_att_trainval.yaml', resume='ckpt'):
        self.options = {
            'logs': {
                'dir_logs': dir_logs
            }
        }
        with open(config, 'r') as handle:
            options_yaml = yaml.load(handle)
        self.options = utils.update_values(self.options, options_yaml)
        
        self.trainset = datasets.factory_VQA(self.options['vqa']['trainsplit'],
                                        self.options['vqa'])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(self.options['coco']['size']),
            transforms.CenterCrop(self.options['coco']['size']),
            transforms.ToTensor(),
            normalize,
        ])

        opt_factory_cnn = {
            'arch': self.options['coco']['arch']
            }
        self.cnn = convnets.factory(opt_factory_cnn, cuda=torch.cuda.is_available(), data_parallel=False)

        self.model = models.factory(
            self.options['model'],
            self.trainset.vocab_words(),
            self.trainset.vocab_answers(),
            cuda=torch.cuda.is_available(),
            data_parallel=False
            )
        start_epoch, best_acc1, _ = load_checkpoint(self.model, None,
            os.path.join(self.options['logs']['dir_logs'], resume))
        self.classes = classes_idxs
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        with torch.no_grad():
            self.model.seq2vec.register_forward_hook(get_activation('seq2vec'))
            self.model.conv_att.register_forward_hook(get_activation('conv_att'))
        self.activation = activation

    def interpolate(self, img, heat_map):
        height = img.shape[0]
        width = img.shape[1]

        # resize heat map
        heat_map_resized = transform.resize(heat_map, (height, width,1))
        # normalize
        max_value = np.max(heat_map_resized)
        min_value = np.min(heat_map_resized)
        normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)
        return normalized_heat_map

    def grad_cam(self, img, logits, activations):
        img = img.permute(1,2,0)
        relu = torch.nn.ReLU()
        pred = logits.argmax(dim=1)
        logits[:,pred.item()].backward()
        gradients = self.model.conv_att.weight.grad
        pooled_gradients = torch.mean(gradients, dim=[0,2,3])
        for i in range(2):
            activations[:, i, :, :] *= pooled_gradients[i]
        activations = relu(activations)

        heat = torch.mean(activations, dim=1).squeeze()
        heat = ndimage.filters.gaussian_filter(heat.cpu(), sigma=2)

        heat = self.interpolate(np.array(img), 1 * heat)
        heat[heat > 0.95] = 0.95
        heat[heat < 0.2] = 0.2
        heat = torch.Tensor(heat)
        fg_im = (heat * img).permute(2,0,1)
        bg_im = ((1.-heat) * img).permute(2,0,1)
        
        return heat, fg_im[None,:,:], bg_im[None,:,:]
    
    def infer(self, img, question):
        """
        :param img: PIL image object
        :param question (str): 

        Returns:
            The top five answers, the final logits weight vector, and the question embedding
        """
        with torch.no_grad():
            v = process_visual(img, self.transform, self.options, self.cnn)
            q = process_question(question, self.trainset)
        
        # get the output after the first layer
        
        logits = self.model(v,q) # logit weight vector of the answers
        a = process_answer(logits, self.trainset, self.model)

        #logits_cls = torch.clone(logits)
        #logits_cls = logits_cls.cpu().detach().numpy()[:,self.classes]
        #logits_cls = torch.Tensor(logits_cls).to("cuda")
        q_emb = self.activation['seq2vec']
        att_activation = self.activation['conv_att']
        
        del v,q
        return a, logits, q_emb, att_activation
