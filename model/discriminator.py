import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from spectral_norm import SpectralNorm

class CNNBlock(nn.Module):
    """
    Class to initialize a convolution block.

    Parameters
    ----------
        in_channels : integer
            Number of channels in the input image.
        out_channels : integer
            Number of channels produced by the convolution block.
        strid : integer
            Stride of the convolution.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect")
        
        self.spec = nn.Sequential(
            SpectralNorm(self.conv),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.spec(x)


class Pix2PixDiscriminator(nn.Module):
    """
    PatchGAN discriminator for CountEx-VQA

    Parameters
    ----------
        in_channels : integer
            Number of channels in the input image.
        features : [integer], optional
            Number of features output by the four convolution blocks.
            The default value is [64,128,256,512].
        ans_logits_dim : integer, optional
            The dimension of the VQA model's final logits weight vector.
            The default value is 414.
        img_size : integer, optional
            Height and width of the input image.
            The default value is 256.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], ans_logits_dim=414, img_size=256):
        super().__init__()
        
        self.pre_embed = nn.Linear(ans_logits_dim, img_size*img_size)
        self.embed = SpectralNorm(self.pre_embed)
        
        self.initial = nn.Conv2d(
                in_channels+1,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, logits):
        ans_embedding = self.embed(1 * logits).view(logits.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, ans_embedding], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    