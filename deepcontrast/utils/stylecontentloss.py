
"""
Neural Style Transfer Tutorial by PyTorch
Link: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
Image 1: https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg
Image 2: https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg
"""

### IMPORT PACKAGES HERE ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from PIL import Image

import torchvision.transforms as transforms


import copy
import numpy as np

### Introduce global functions here ###

def gram_matrix(input): #input is a matrix containing the feature maps of image X at layer L, F_{XL}
	a, b, c, d = input.size() # a=batch size, b=number of maps, c and d = dims
	N = a*b*c*d

	features = input.view(a*b, c*d)
	G = torch.mm(features, features.t()) # matrix multiply features with its transpose

	return G.div(N) #normalize by number of elements because the larger the N, the larger the elements


### Content loss as a transparent layer ###

class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach() # detach to avoid computing the gradient. If you don't do this, apparently the forward method will throw an error? Look into this

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target) # apparently no difference between F.mse_loss and nn.MSE_loss
		return input # return the input to make this layer transparent



### Style loss as a transparent layer ###

class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()

	def forward(self, input):
		self.loss = F.mse_loss(gram_matrix(input), self.target)
		return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        #self.mean = torch.tensor(mean).view(-1, 1, 1)
        #self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)


    def forward(self, img):
        # normalize ``img``
        return (img - self.mean.to(img.device)) / self.std.to(img.device)


def get_style_content_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,input_img,
                               content_layers,
                               style_layers,style_weight,content_weight):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            print("style layer name is ", name)
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    model(input_img)
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    print("scloss is", loss)

    return loss














