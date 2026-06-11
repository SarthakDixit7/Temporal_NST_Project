from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


def gram_matrix(feature_map):
    batch_size, channels, height, width = feature_map.shape
    features = feature_map.view(batch_size, channels, height * width)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features, features_t)
    return gram / (channels * height * width)


def _load_vgg19_features():
    try:
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features
    except Exception:
        vgg = models.vgg19(pretrained=True).features

    for idx, layer in enumerate(vgg):
        if isinstance(layer, nn.MaxPool2d):
            vgg[idx] = nn.AvgPool2d(kernel_size=2, stride=2)
    return vgg


class VGG19LossNetwork(nn.Module):
    """Frozen VGG-19 feature extractor for perceptual and style losses."""

    def __init__(self):
        super().__init__()
        self.vgg = _load_vgg19_features()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()

        self.layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
        }

    def forward(self, x):
        content_feature = None
        style_features = OrderedDict()

        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx == self.layer_map["relu1_2"]:
                style_features["relu1_2"] = x
            elif idx == self.layer_map["relu2_2"]:
                style_features["relu2_2"] = x
            elif idx == self.layer_map["relu3_3"]:
                content_feature = x
                style_features["relu3_3"] = x
            elif idx == self.layer_map["relu4_3"]:
                style_features["relu4_3"] = x
                break

        return content_feature, style_features


class VGGFeatureExtractor(nn.Module):
    """Thin wrapper for extracting content/style features and Gram matrices."""

    def __init__(self):
        super().__init__()
        self.net = VGG19LossNetwork()

    def forward(self, x):
        content_feature, style_features = self.net(x)
        style_grams = {name: gram_matrix(feature) for name, feature in style_features.items()}
        return content_feature, style_features, style_grams
