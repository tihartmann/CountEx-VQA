# Pretrained models for Pytorch (Work in progress)

The goal of this repo is:

- to help to reproduce research papers results (transfer learning setups for instance),
- to access pretrained ConvNets with a unique interface/API inspired by torchvision.

News:

- 08/12/2017: update data url (/!\ `git pull` is needed)
- 30/11/2017: improve API (`model.features(input)`, `model.logits(features)`, `model.forward(input)`, `model.last_linear`)
- 16/11/2017: nasnet-a-large pretrained model ported by T. Durand and R. Cadene
- 22/07/2017: torchvision pretrained models
- 22/07/2017: momentum in inceptionv4 and inceptionresnetv2 to 0.1
- 17/07/2017: model.input_range attribut
- 17/07/2017: BNInception pretrained on Imagenet

## Summary

- [Installation](https://github.com/Cadene/pretrained-models.pytorch#installation)
- [Quick examples](https://github.com/Cadene/pretrained-models.pytorch#quick-examples)
- [Few use cases](https://github.com/Cadene/pretrained-models.pytorch#few-use-cases)
    - [Compute imagenet logits](https://github.com/Cadene/pretrained-models.pytorch#compute-imagenet-logits)
    - [Compute imagenet validation metrics](https://github.com/Cadene/pretrained-models.pytorch#compute-imagenet-validation-metrics)
- [Evaluation on ImageNet](https://github.com/Cadene/pretrained-models.pytorch#evaluation-on-imagenet)
    - [Accuracy on valset](https://github.com/Cadene/pretrained-models.pytorch#accuracy-on-validation-set)
    - [Reproducing results](https://github.com/Cadene/pretrained-models.pytorch#reproducing-results)
- [Documentation](https://github.com/Cadene/pretrained-models.pytorch#documentation)
    - [Available models](https://github.com/Cadene/pretrained-models.pytorch#available-models)
        - [AlexNet](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception)
        - [DenseNet121](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [DenseNet161](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [DenseNet169](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [DenseNet201](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet)
        - [InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception)
        - [InceptionV3](https://github.com/Cadene/pretrained-models.pytorch#inception)
        - [InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception)
        - [NASNet-A-Large](https://github.com/Cadene/pretrained-models.pytorch#nasnet)
        - [ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext)
        - [ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext)
        - [ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [ResNet152](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [SqueezeNet1_0](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [SqueezeNet1_1](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG11](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG13](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG16](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG19](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG11_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG13_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG16_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
        - [VGG19_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision)
    - [Model API](https://github.com/Cadene/pretrained-models.pytorch#model-api)
        - [model.input_size](https://github.com/Cadene/pretrained-models.pytorch#modelinput_size)
        - [model.input_space](https://github.com/Cadene/pretrained-models.pytorch#modelinput_space)
        - [model.input_range](https://github.com/Cadene/pretrained-models.pytorch#modelinput_range)
        - [model.mean](https://github.com/Cadene/pretrained-models.pytorch#modelmean)
        - [model.std](https://github.com/Cadene/pretrained-models.pytorch#modelstd)
        - [model.features](https://github.com/Cadene/pretrained-models.pytorch#modelfeatures)
        - [model.logits](https://github.com/Cadene/pretrained-models.pytorch#modellogits)
        - [model.forward](https://github.com/Cadene/pretrained-models.pytorch#modelforward)
- [Reproducing porting](https://github.com/Cadene/pretrained-models.pytorch#reproducing)
    - [ResNet*](https://github.com/Cadene/pretrained-models.pytorch#hand-porting-of-resnet152)
    - [ResNeXt*](https://github.com/Cadene/pretrained-models.pytorch#automatic-porting-of-resnext)
    - [Inception*](https://github.com/Cadene/pretrained-models.pytorch#hand-porting-of-inceptionv4-and-inceptionresnetv2)

## Installation

1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)
3. `git clone https://github.com/Cadene/pretrained-models.pytorch.git`


## Quick examples

- To import `pretrainedmodels`:

```python
import sys
sys.path.append('yourdir/pretrained-models.pytorch') # if needed
import pretrainedmodels
```

- To print the available pretrained models:

```python
model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))
print(model_names)
```

- To load a pretrained models from imagenet:

```python
model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
```

**Note**: By default, models will be downloaded to your `$HOME/.torch` folder. You can modify this behavior using the `$TORCH_MODEL_ZOO` variable as follow: `export TORCH_MODEL_ZOO="/local/pretrainedmodels`

- To load an image and do a complete forward pass:

```python
import torch
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

# transformations depending on the model
#??rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

path_img = 'data/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)

output_logits = model(input) # 1x1000
```

- To extract features (beware this API is not available for all networks):

```python
output_features = model.features(input) # 1x14x14x2048 size may differ
output_logits = model.logits(output_features) # 1x1000
```

## Few use cases

### Compute imagenet logits

- See [examples/imagenet_logits.py](https://github.com/Cadene/pretrained-models.pytorch/blob/master/examples/imagenet_logits.py) to compute logits of classes appearance over a single image with a pretrained model on imagenet.

```
$ python examples/imagenet_logits.py -h
> nasnetalarge, resnet152, inceptionresnetv2, inceptionv4, ...
```

```
$ python examples/imagenet_logits.py -a nasnetalarge --path_img data/cat.png
> 'nasnetalarge': data/cat.png' is a 'tiger cat' 
```

### Compute imagenet evaluation metrics

- See [examples/imagenet_eval.py](https://github.com/Cadene/pretrained-models.pytorch/blob/master/examples/imagenet_eval.py) to evaluate pretrained models on imagenet valset. 

```
$ python examples/imagenet_eval.py /local/common-data/imagenet_2012/images -a nasnetalarge -b 20 -e
> * Acc@1 92.693, Acc@5 96.13
```


## Evaluation on imagenet

### Accuracy on validation set (single model)

Model | Version | Acc@1 | Acc@5
--- | --- | --- | ---
NASNet-A-Large | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 82.693 | 96.163
[NASNet-A-Large](https://github.com/Cadene/pretrained-models.pytorch#nasnet) | Our porting | 82.566 | 96.086
InceptionResNetV2 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.4 | 95.3
InceptionV4 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.2 | 95.3
[InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting | 80.170 | 95.234
[InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting | 80.062 | 94.926
ResNeXt101_64x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 79.6 | 94.7
[ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting | 78.956 | 94.252
ResNeXt101_32x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 78.8 | 94.4
ResNet152 | [Pytorch](https://github.com/pytorch/vision#models) | 78.428 | 94.110
[ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting | 78.188 | 93.886
FBResNet152 | [Torch7](https://github.com/facebook/fb.resnet.torch) | 77.84 | 93.84
[DenseNet161](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.560 | 93.798
[FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet) | Our porting | 77.386 | 93.594
[InceptionV3](https://github.com/Cadene/pretrained-models.pytorch#inception) | [Pytorch](https://github.com/pytorch/vision#models) | 77.294 | 93.454
[DenseNet201](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.152 | 93.548
[ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.438 | 93.672
[DenseNet169](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 76.026 | 92.992
[ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 76.002 | 92.980
[DenseNet121](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 74.646 | 92.136
[VGG19_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 74.266 | 92.066
[ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 73.554 | 91.456
[BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception) | Our porting | 73.522 | 91.560
[VGG16_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 73.518 | 91.608
[VGG19](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 72.080 | 90.822
[VGG16](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 71.636 | 90.354
[VGG13_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 71.508 | 90.494
[VGG11_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 70.452 | 89.818
[ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 70.142 | 89.274
[VGG13](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 69.662 | 89.264
[VGG11](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 68.970 | 88.746
[SqueezeNet1_1](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 58.250 | 80.800
[SqueezeNet1_0](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 58.108 | 80.428
[Alexnet](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 56.432 | 79.194

Note: the Pytorch version of ResNet152 is not a porting of the Torch7 but has been retrained by facebook.

Beware, the accuracy reported here is not always representative of the transferable capacity of the network on other tasks and datasets. You must try them all! :P
    
### Reproducing results

Please see [Compute imagenet validation metrics](https://github.com/Cadene/pretrained-models.pytorch#compute-imagenet-validation-metrics)


## Documentation

### Available models

#### NASNet*

Source: [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/slim)

- `nasnetalarge(num_classes=1000, pretrained='imagenet')`
- `nasnetalarge(num_classes=1001, pretrained='imagenet+background')`

#### FaceBook ResNet*

Source: [Torch7 repo of FaceBook](https://github.com/facebook/fb.resnet.torch)

There are a bit different from the ResNet* of torchvision. ResNet152 is currently the only one available.

- `fbresnet152(num_classes=1000, pretrained='imagenet')`

#### Inception*

Source: [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/slim) and [Pytorch/Vision repo](https://github.com/pytorch/vision/tree/master/torchvision) for `inceptionv3`

- `inceptionresnetv2(num_classes=1000, pretrained='imagenet')`
- `inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')`
- `inceptionv4(num_classes=1000, pretrained='imagenet')`
- `inceptionv4(num_classes=1001, pretrained='imagenet+background')`
- `inceptionv3(num_classes=1000, pretrained='imagenet')`

#### BNInception

Source: [Trained with Caffe](https://github.com/Cadene/tensorflow-model-zoo.torch/pull/2) by [Xiong Yuanjun](http://yjxiong.me)

- `bninception(num_classes=1000, pretrained='imagenet')`

#### ResNeXt*

Source: [ResNeXt repo of FaceBook](https://github.com/facebookresearch/ResNeXt)

- `resnext101_32x4d(num_classes=1000, pretrained='imagenet')`
- `resnext101_62x4d(num_classes=1000, pretrained='imagenet')`

#### TorchVision
Source: [Pytorch/Vision repo](https://github.com/pytorch/vision/tree/master/torchvision)

(`inceptionv3` included in [Inception*](https://github.com/Cadene/pretrained-models.pytorch#inception))

- `resnet18(num_classes=1000, pretrained='imagenet')`
- `resnet34(num_classes=1000, pretrained='imagenet')`
- `resnet50(num_classes=1000, pretrained='imagenet')`
- `resnet101(num_classes=1000, pretrained='imagenet')`
- `resnet152(num_classes=1000, pretrained='imagenet')`
- `densenet121(num_classes=1000, pretrained='imagenet')`
- `densenet161(num_classes=1000, pretrained='imagenet')`
- `densenet169(num_classes=1000, pretrained='imagenet')`
- `densenet201(num_classes=1000, pretrained='imagenet')`
- `squeezenet1_0(num_classes=1000, pretrained='imagenet')`
- `squeezenet1_1(num_classes=1000, pretrained='imagenet')`
- `alexnet(num_classes=1000, pretrained='imagenet')`
- `vgg11(num_classes=1000, pretrained='imagenet')`
- `vgg13(num_classes=1000, pretrained='imagenet')`
- `vgg16(num_classes=1000, pretrained='imagenet')`
- `vgg19(num_classes=1000, pretrained='imagenet')`
- `vgg11_bn(num_classes=1000, pretrained='imagenet')`
- `vgg13_bn(num_classes=1000, pretrained='imagenet')`
- `vgg16_bn(num_classes=1000, pretrained='imagenet')`
- `vgg19_bn(num_classes=1000, pretrained='imagenet')`


### Model API

Once a pretrained model has been loaded, you can use it that way.

**Important note**: All image must be loaded using `PIL` which scales the pixel values between 0 and 1.

#### `model.input_size`

Attribut of type `list` composed of 3 numbers:

- number of color channels,
- height of the input image,
- width of the input image.

Example:

- `[3, 299, 299]` for inception* networks,
- `[3, 224, 224]` for resnet* networks.


#### `model.input_space`

Attribut of type `str` representating the color space of the image. Can be `RGB` or `BGR`.


#### `model.input_range`

Attribut of type `list` composed of 2 numbers:

- min pixel value,
- max pixel value.

Example:

- `[0, 1]` for resnet* and inception* networks,
- `[0, 255]` for bninception network.


#### `model.mean`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (substract "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.485, 0.456, 0.406]` for resnet* networks.


#### `model.std`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (divide "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.229, 0.224, 0.225]` for resnet* networks.


#### `model.features`

/!\ work in progress (may not be available)

Method which is used to extract the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
print(input_224.size())            # (1,3,224,224)
output = model.features(input_224) 
print(output.size())               # (1,2048,1,1)

# print(input_448.size())          # (1,3,448,448)
output = model.features(input_448)
# print(output.size())             # (1,2048,7,7)
```

#### `model.logits`

/!\ work in progress (may not be available)

Method which is used to classify the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
output = model.features(input_224) 
print(output.size())               # (1,2048, 1, 1)
output = model.logits(output)
print(output.size())               # (1,1000)
```

#### `model.forward`

Method used to call `model.features` and `model.logits`. It can be overwritten as desired.

**Note**: A good practice is to use `model.__call__` as your function of choice to forward an input to your model. See the example bellow.

```python
# Without model.__call__
output = model.forward(input_224)
print(output.size())      # (1,1000)

# With model.__call__
output = model(input_224)
print(output.size())      # (1,1000)
```

#### `model.last_linear`

Attribut of type `nn.Linear`. This module is the last one to be called during the forward pass.

- Can be replaced by an adapted `nn.Linear` for fine tuning.
- Can be replaced by `pretrained.utils.Identity` for features extraction. 

Example when the model is loaded using `fbresnet152`:

```python
print(input_224.size())            # (1,3,224,224)
output = model.features(input_224) 
print(output.size())               # (1,2048,1,1)
output = model.logits(output)
print(output.size())               # (1,1000)

# fine tuning
dim_feats = model.last_linear.in_features # =2048
nb_classes = 4
model.last_linear = nn.Linear(dim_feats, nb_classes)
output = model(input_224)
print(output.size())               # (1,4)

# features extraction
model.last_linear = pretrained.utils.Identity()
output = model(input_224)
print(output.size())               # (1,2048)
```

## Reproducing

### Hand porting of ResNet152

```
th pretrainedmodels/fbresnet/resnet152_dump.lua
python pretrainedmodels/fbresnet/resnet152_load.py
```

### Automatic porting of ResNeXt

https://github.com/clcarwin/convert_torch_to_pytorch

### Hand porting of NASNet, InceptionV4 and InceptionResNetV2

https://github.com/Cadene/tensorflow-model-zoo.torch


## Acknowledgement

Thanks to the deep learning community and especially to the contributers of the pytorch ecosystem.