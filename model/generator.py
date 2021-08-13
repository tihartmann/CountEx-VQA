import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_norm import SpectralNorm

class Block(nn.Module):
    """
    A class for convolution / deconvolution operations.
    Performs 2D (de)convolution with instance normalization and (leaky) ReLU activation function.
    For deconvolution, spectral normalization is applied in addition.    

    Parameters:
    ----------
        in_channels : integer
            Number of channels in the input image.
        out_channels : integer
            Number of channels produced by the convolution.
        down : boolean, optional
            If True, the Block is initialized as a convolution layer.
            If False, the Block is initialized as a deconvolution layer.
            The default value is True.
        act : string, optional
            Specifies whether the Block should use a ReLU or a leaky ReLU activation function.
            Has to be in ["relu", "leaky"].
            The default value is "relu".
        use_dropout : boolean, optional
            If True, the block applies Dropout.
            The default value is False.
    """
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        if down:
            self.full_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            self.full_conv = nn.Sequential(
                SpectralNorm(self.conv),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )  

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.full_conv(x)
        
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    """
    Generates counterfactual images.

    Parameters
    ----------
        in_channels : integer
            Number of channels of the input image.
        features : integer, optional
            Number of features output by the convolution blocks. 
            The default value is 64.
        q_emb_dim : integer, optional
            Dimension of the question embedding.
            The default value is 2400.
        ans_logits_dim : integer, optional
            Dimension of the VQA model's logits weight vector.
            The default value is 414.
    """
    def __init__(self, in_channels=3, features=64, q_emb_dim=2400, ans_logits_dim=414):
        super(Generator, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels+1, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1 = Block(features, features, down=True, act="leaky", use_dropout=True)
        self.down2 = Block(features, features, down=True, act="leaky", use_dropout=True)
        self.down3 = Block(features, features, down=True, act="leaky", use_dropout=True)
        
        self.sliced_vector_size = (q_emb_dim + ans_logits_dim) // 3
        flattened_conv_filter_size = 1 * 1 * features * features
        self.text2conv1 = nn.Linear(self.sliced_vector_size, flattened_conv_filter_size)
        #self.norm1 = nn.InstanceNorm2d(features)
        self.text2conv2 = nn.Linear(self.sliced_vector_size, flattened_conv_filter_size)
        #self.norm2 = nn.InstanceNorm2d(features)
        self.text2conv3 = nn.Linear(self.sliced_vector_size, flattened_conv_filter_size)
        #self.norm3 = nn.InstanceNorm2d(features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features, features, 4, 2, 1),
        )
        
        self.ans2conv = nn.Linear(ans_logits_dim, (64))
        
        self.up1 = Block(features+1, features, down=False, act="leaky", use_dropout=True) #changed from relu
        self.up2 = Block(features*2, features, down=False, act="leaky", use_dropout=True)#changed from relu
        self.up3 = Block(features*2, features, down=False, act="leaky", use_dropout=False)#changed from relu
        self.up4 = Block(features*2, features, down=False, act="leaky", use_dropout=False)#changed from relu
        self.final_up_pre = nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1)
        self.final_up = nn.Sequential(
            #nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            SpectralNorm(self.final_up_pre),
            nn.Tanh(),
        )
        
    def forward(self, x, q_emb, logits):
        batch_size, img_channels, height, width = x.size()
        q_emb = torch.cat([q_emb, logits],1)
        
        # first down
        ################################################
        d1 = self.initial_down(x)
        ###############################################
        
        # second down
        ################################################
        d2 = self.down1(d1)
        text_slice = q_emb[:,: self.sliced_vector_size]
        conv_kernel_shape=(batch_size, d2.shape[1], d2.shape[1], 1, 1)
        text_conv_filters = self.text2conv1(text_slice).view(conv_kernel_shape)
        text_conv_filters = text_conv_filters.view(-1, *text_conv_filters.size()[2:])
        d2 = F.conv2d(d2, text_conv_filters, groups=d2.size()[0]).view(d2.size())
        ################################################
        
        # third down
        ################################################
        d3 = self.down2(d2)
        text_slice = q_emb[:, self.sliced_vector_size : self.sliced_vector_size * 2]
        conv_kernel_shape=(batch_size, d3.shape[1], d3.shape[1], 1, 1)
        text_conv_filters = self.text2conv2(text_slice).view(conv_kernel_shape)
        text_conv_filters = text_conv_filters.view(-1, *text_conv_filters.size()[2:])
        d3 = F.conv2d(d3, text_conv_filters, groups=d3.size()[0]).view(d3.size())
        ################################################
        
        # fourth down
        ################################################
        d4 = self.down3(d3)
        text_slice = q_emb[:, self.sliced_vector_size * 2:]
        conv_kernel_shape=(batch_size, d4.shape[1], d4.shape[1], 1, 1)
        text_conv_filters = self.text2conv3(text_slice).view(conv_kernel_shape)
        text_conv_filters = text_conv_filters.view(-1, *text_conv_filters.size()[2:])
        d4 = F.conv2d(d4, text_conv_filters, groups=d4.size()[0]).view(d4.size())
        ################################################
        
        # bottleneck
        bottleneck = self.bottleneck(d4)
        ans_embedding = self.ans2conv(1 * logits).view(batch_size, 1, 8, 8)
        bottleneck = torch.cat([bottleneck, ans_embedding], 1)

        # upsampling
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d4],1))
        up3 = self.up3(torch.cat([up2,d3], 1))
        up4 = self.up4(torch.cat([up3,d2], 1))
        up5 = self.final_up(torch.cat([up4,d1], 1))
        return up5  

if __name__ == "__main__":
    g = Generator()
    img = torch.randn((1,3,256,256))
    q_emb = torch.randn((1,2400))
    ans_emb = torch.randn((1,414))
    att = torch.randn((1,1,256,256))

    preds = g(torch.cat([img,att],1), q_emb, ans_emb)
    print(preds.shape)