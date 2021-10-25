#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

# model.
import timm

# custom modules.
from . import vision_transformer as vits
from .utils import load_pretrained_weights, load_pretrained_weights_resnet

class NishikaCustomModelClass(nn.Module):
    def __init__(self, architecture):
        super(NishikaCustomModelClass, self).__init__()

        self.model = timm.create_model(architecture, pretrained=True, in_chans=3)
        #print(self.model)

        if 'vit' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, 15)
        elif 'deit' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, 15)
        elif 'swin' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, 15)
        elif 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, 15)
        elif 'efficient' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 15)
        elif 'ensenet' in architecture:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 15)
        elif 'nfnet' in architecture:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(self.n_features, 15)

    def forward(self, x):
        x = self.model(x)
        return x
        
class NishikaCustomModelClass_USBmicroBW(nn.Module):
    def __init__(self, architecture):
        super(NishikaCustomModelClass_USBmicroBW, self).__init__()
        self.model = timm.create_model(architecture, pretrained=True, in_chans=3)
        if 'vit' in architecture:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Linear(self.n_features, 2)
        elif 'resnet' in architecture:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.n_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x