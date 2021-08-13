**Note: This repository is still under development and will be finalized soon!**

# CountEx-VQA - Counterfactual Explanations for Visual Question Answering
This repository contains the code for my Master's thesis "Generating Counterfactual Images for VQA by Editing Question Critical Objects" at WeST, University of Koblenz-Landau.
The thesis was supervised by Prof. Dr. Matthias Thimm and Dr. Zeyd Boukhers. This repo was made by [Timo Hartmann](https://www.researchgate.net/profile/Timo-Hartmann-3). 

We developed this code to generate counterfactual images for a VQA model to increase its interpretability. Specifically, the proposed method, *CountEx-VQA*, uses a Generative Adversarial Network to translate an image into a minimally different counterfactual such that the prediction of a VQA model based on a given question changes. 

## Installation

### Requirements
The code in this repository requires Python 3. We advise you to run the code inside an Anaconda environment:

```
conda create --name countex-vqa python=3.8
source activate countex-vqa
```

Next, clone the repository and install the requirements:

```
cd $HOME
git clone https://github.com/tihartmann/CountEx-VQA.git
cd CountEx-VQA
pip install -r requirements.txt
```
### Data
The data used in the thesis' experiments and the pre-trained MUTAN VQA model can be downloaded as follows:
```
cd tools/
./download.sh

```


## Reproduce Results
To reproduce the results from the experiments described in the thesis, run the following code. Note that, by default, the training script uses CUDA if available. 
If you want to manually specify the device, you can do so by changing line 3 in the ```config.py``` file. 

```
cd $HOME/CountEx-VQA
python train.py
```

If you want to start the training procedure using the pre-trained CountEx-VQA model, download the weights using the link below and set ```LOAD_MODEL = TRUE``` in line 15 of ```config.py```.

### Pretrained Models
The pretrained generator and discriminator can be downloaded [here](https://drive.google.com/drive/folders/1i5D6dDg-6wAiToT4QnlxIC5oElVc5XyF?usp=sharing).

## Web Demo

## Acknowledgement
This project uses a pre-trained MUTAN VQA-Model as described by [Ben-younes et al.](https://arxiv.org/abs/1705.06676).
The code and pre-trained models can be found in [this repository](https://github.com/Cadene/vqa.pytorch)
