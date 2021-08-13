import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMAGE_DIR = "../data/coco/train"
VAL_IMAGE_DIR = "../data/coco/val"
D_LEARNING_RATE = 2e-4 # original 2e-4
G_LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 60
CE_LAMBDA = [100,30]
L2_LAMBDA = [1, 0.03]
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "discriminator.pth.tar"
CHECKPOINT_GEN = "generator.pth.tar"
