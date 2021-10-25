#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import f1_score

class CustomLossClassification:
    def __init__(self):
        self.loss_cls = nn.CrossEntropyLoss()

    def calc_loss(self, preds, targets):
        return self.loss_cls(preds, targets)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']
        #print(targets.dtype, targets)
        loss = lam*self.loss_cls(preds, targets) + (1-lam)*self.loss_cls(preds, shuffled_targets)
        return loss

    def calc_metrics_f1(self, preds, targets):
        loss = f1_score(targets, preds, average='micro')
        return loss

    def calc_metrics_accuracy(self, preds, targets):
        loss = accuracy_score(targets, preds)
        return loss

class CustomLossClassification_USBmicroBW:
    def __init__(self):
        #self.loss_cls = nn.CrossEntropyLoss()
        self.loss_cls = nn.BCELoss()

    def calc_loss(self, preds, targets):
        return self.loss_cls(preds, targets)

    def calc_mixloss(self, preds, results):
        targets = results['targets']
        shuffled_targets = results['shuffled_targets']
        lam = results['lam']
        #print(targets.dtype, targets)
        loss = lam*self.loss_cls(preds, targets) + (1-lam)*self.loss_cls(preds, shuffled_targets)
        return loss

    def calc_metrics_f1(self, preds, targets):
        loss = f1_score(targets, preds, average='micro')
        return loss

    def calc_metrics_accuracy(self, preds, targets):
        loss = accuracy_score(targets, preds)
        return loss