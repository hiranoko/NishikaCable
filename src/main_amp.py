#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler 
from torchvision.transforms import Normalize
from albumentations import Compose, OneOf, ShiftScaleRotate, HorizontalFlip, Resize, Transpose
from albumentations.augmentations import transforms as aug
from albumentations.pytorch.transforms import ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
import random
from pathlib import Path
import gc

# utils.
from utils import util
from utils.config import Config
import factory
from utils.cutmix_mixup import cutmix, mixup, cutmix_soft_label, mixup_soft_label

# Third party
import ttach as tta
from optimizer.sam import SAM

class Runner:
    def __init__(self, cfg, model, criterion, optimizer, scheduler, device, logger):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger

    def train(self, dataset_trn, dataset_val, loader_trn, loader_val):
        print(f'training start at {datetime.datetime.now()}')

        self.cfg.snapshot.output_dir = self.cfg.working_dir / 'weight'
        snap = util.Snapshot(**self.cfg.snapshot)

        self.use_mixup = self.cfg.use_mixup_cutmix

        for epoch in range(self.cfg.n_epochs):
            start_time = time.time()

            if epoch >= self.cfg.wo_mixup_epochs:
                self.use_mixup = False

            # train.
            if self.cfg.sam_optimizer:
                result_trn = self.run_nn_sam('trn', dataset_trn, loader_trn)
            else:
                result_trn = self.run_nn('trn', dataset_trn, loader_trn)

            # valid.
            with torch.no_grad():
                result_val = self.run_nn('val', dataset_val, loader_val)

            # scheduler step.
            if self.scheduler.__class__.__name__=='ReduceLROnPlateau':
                self.scheduler.step(result_val[self.cfg.scheduler.monitor])
            else:
                self.scheduler.step()

            wrap_time = time.time()-start_time

            # logging.
            logging_info = [epoch+1, wrap_time]
            logging_info.extend(sum([[result_trn[i], result_val[i]] for i in self.cfg.logger.params.logging_info], []))
            if self.logger:
                self.logger.write_log(logging_info)

            # metrics
            print(f"{epoch+1}/{self.cfg.n_epochs}: trn_loss={result_trn['loss']:.4f}, val_loss={result_val['loss']:.4f}, trn_metric={result_trn['metric']:.4f}, val_metric={result_val['metric']:.4f}, time={wrap_time:.2f}sec")

            # snapshot.
            snap.snapshot(result_val[self.cfg.snapshot.monitor], self.model, self.optimizer, epoch)

    def test(self, dataset_test, loader_test):
        print(f'test start at {datetime.datetime.now()}')
        with torch.no_grad():
            result = self.run_nn('test', dataset_test, loader_test)
        print('done.')
        return result

    def run_nn(self, mode, dataset, loader):

        losses = []
        metrics = 0
        raw_pred = np.zeros((len(dataset), self.cfg.n_class))
        raw_target = np.zeros(len(dataset))
        
        if self.cfg.use_amp:
            scaler = GradScaler()

        if mode=='trn':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode=='val' or mode=='test':
            self.model.eval()
        
        for idx, batch in enumerate(tqdm(loader)):
            img    = batch['image']
            target = batch['target']

            img   = img.to(self.device, dtype=torch.float)
            label = target.to(self.device, dtype=torch.long)

            if mode=='trn' and self.use_mixup:
                p = np.random.rand()
                if 0.0<=p<self.cfg.mixup_freq/2:
                    # apply mixup.
                    img, results = mixup(img, label, self.cfg.mixup_alpha)
                    if self.cfg.use_amp:
                        with autocast():
                            pred = self.model(img)
                            loss = self.criterion.calc_mixloss(pred, results)
                    else:
                        pred = self.model(img)
                        loss = self.criterion.calc_mixloss(pred, results)
                elif self.cfg.mixup_freq/2<=p<self.cfg.mixup_freq: # note use.
                    # apply cutmix.
                    img, results = cutmix(img, label, self.cfg.cutmix_alpha)
                    if self.cfg.use_amp:
                        with autocast():
                            pred = self.model(img)
                            loss = self.criterion.calc_mixloss(pred, results)
                    else:
                        pred = self.model(img)
                        loss = self.criterion.calc_mixloss(pred, results)

                else:
                    # wo mixup cutmix.
                    if self.cfg.use_amp:
                        with autocast():
                            pred = self.model(img)
                            loss = self.criterion.calc_loss(pred, label)
                    else:
                        pred = self.model(img)
                        loss = self.criterion.calc_loss(pred, label)
            else:
                # pred and calc losses.
                if self.cfg.use_amp:
                    with autocast():
                        pred = self.model(img)
                        loss = self.criterion.calc_loss(pred, label)
                else:
                    pred = self.model(img)
                    loss = self.criterion.calc_loss(pred, label)
                
            losses.append(loss.item())

            if mode=='trn':
                if self.cfg.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # make predictions.
            raw_pred[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size] += pred.softmax(1).detach().cpu().squeeze().numpy().reshape(-1, self.cfg.n_class)
            # targets.
            raw_target[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size] += label.detach().cpu().squeeze().numpy()

        # calc metrics
        if mode=='trn' or mode=='val':
            _raw_pred = np.argmax(raw_pred, axis=1)
            metrics = self.criterion.calc_metrics_f1(_raw_pred, raw_target)
        elif mode=='test':
            return raw_pred

        result = dict(
            loss=np.sum(losses)/len(loader),
            metric=metrics,
        )

        return result

    def run_nn_sam(self, mode, dataset, loader):

        losses = []
        metrics = 0
        raw_pred = np.zeros((len(dataset), self.cfg.n_class))
        raw_target = np.zeros(len(dataset))
        
        if self.cfg.use_amp:
            scaler = GradScaler()

        if mode=='trn':
            self.model.train()
            self.optimizer.zero_grad()
        elif mode=='val' or mode=='test':
            self.model.eval()
        
        for idx, batch in enumerate(tqdm(loader)):
            img    = batch['image']
            target = batch['target']

            img   = img.to(self.device, dtype=torch.float)
            label = target.to(self.device, dtype=torch.long)

            if mode=='trn' and self.use_mixup:
                p = np.random.rand()
                if 0.0<=p<self.cfg.mixup_freq/2:
                    # apply mixup.
                    img, results = mixup(img, label, self.cfg.mixup_alpha)
                    # first
                    pred = self.model(img)
                    loss = self.criterion.calc_mixloss(pred, results)
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # second
                    pred = self.model(img)
                    self.criterion.calc_mixloss(pred, results).backward()
                    self.optimizer.second_step(zero_grad=True)
                elif self.cfg.mixup_freq/2<=p<self.cfg.mixup_freq: # note use.
                    # apply cutmix.
                    img, results = cutmix(img, label, self.cfg.cutmix_alpha)
                    # first step
                    pred = self.model(img)
                    loss = self.criterion.calc_mixloss(pred, results)
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # seconda step
                    pred = self.model(img)
                    self.criterion.calc_mixloss(pred, results).backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    # wo mixup cutmix.
                    pred = self.model(img)
                    loss = self.criterion.calc_loss(pred, label)
                    losses.append(loss.item())
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # seconda step
                    pred = self.model(img)
                    self.criterion.calc_loss(pred, label).backward()
                    self.optimizer.second_step(zero_grad=True)
            else:
                # pred and calc losses.
                pred = self.model(img)
                loss = self.criterion.calc_loss(pred, label)
                losses.append(loss.item())
            
            # make predictions.
            raw_pred[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size] += pred.softmax(1).detach().cpu().squeeze().numpy().reshape(-1, self.cfg.n_class)
            # targets.
            raw_target[idx * self.cfg.batch_size:(idx + 1) * self.cfg.batch_size] += label.detach().cpu().squeeze().numpy()

        # calc metrics
        if mode=='trn' or mode=='val':
            _raw_pred = np.argmax(raw_pred, axis=1)
            metrics = self.criterion.calc_metrics_f1(_raw_pred, raw_target)
        elif mode=='test':
            return raw_pred

        result = dict(
            loss=np.sum(losses)/len(loader),
            metric=metrics,
        )

        return result


def train():

    args = util.get_args()
    cfg = Config.fromfile(args.config)
    cfg.fold = args.fold

    cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
    print(f'version: {cfg.version}')
    print(f'fold: {cfg.fold}')

    # make output_dir if needed.
    util.make_output_dir_if_needed(cfg.working_dir)

    # set logger.
    cfg.logger.params.name = cfg.working_dir / f'history.csv'
    my_logger = util.CustomLogger(**cfg.logger.params)

    # set seed.
    util.seed_everything(cfg.seed)

    # get dataloader.
    print(f'dataset: {cfg.dataset_name}')
    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    dataset_train, loader_train = factory.get_dataset_loader(cfg.train, folds)
    dataset_valid, loader_valid = factory.get_dataset_loader(cfg.valid, [cfg.fold])
     
    ##############
    # get model. #
    ##############

    if cfg.tta:
        base_model = factory.get_model(cfg.model)
        model = tta.ClassificationTTAWrapper(
            base_model,
            factory.get_tta_transform(cfg.model)
        )
        print(f'model: TTAmode and {cfg.model.name}')    
    else:
        model = factory.get_model(cfg.model)
        print(f'model: {cfg.model.name}')
    device = factory.get_device(args.gpu)
    model.cuda()
    model.to(device)
    
    ##################
    # get optimizer. #
    ##################

    if cfg.sam_optimizer:
        print(f'optimizer: SAM and {cfg.optim.name}')
        plist = [{'params': model.parameters(), 'lr': cfg.optim.lr, 'weight_decay': cfg.optim.weight_decay}]
        #base_optimizer = factory.get_optimizer(cfg.optim)(plist)
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    else:
        print(f'optimizer: {cfg.optim.name}')
        plist = [{'params': model.parameters(), 'lr': cfg.optim.lr, 'weight_decay': cfg.optim.weight_decay}]
        optimizer = factory.get_optimizer(cfg.optim)(plist)
        

    #############
    # get loss. #
    #############

    print(f'loss: {cfg.loss.name}')
    loss = factory.get_loss(cfg.loss)

    ##################
    # get scheduler. #
    ##################

    print(f'scheduler: {cfg.scheduler.name}')
    scheduler = factory.get_scheduler(cfg.scheduler, optimizer)
 
    ##############
    # run model. #
    ##############

    runner = Runner(cfg, model, loss, optimizer, scheduler, device, my_logger)
    runner.train(dataset_train, dataset_valid, loader_train, loader_valid)

def submission():

    args = util.get_args()
    cfg = Config.fromfile(args.config)
    cfg.fold = args.fold
    cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
    print(f'version: {cfg.version}')
    print(f'fold: {cfg.fold}')

    # set logger.
    my_logger = None

    # set seed.
    util.seed_everything(cfg.seed)

    # get dataloader.
    dataset_test, loader_test = factory.get_dataset_loader(cfg.test, [0])

    ###############
    # get model.  #
    ###############

    print(f'model: {cfg.model.name}')
    if cfg.tta:
        base_model = factory.get_model(cfg.model)
        model = tta.ClassificationTTAWrapper(
            base_model,
            factory.get_tta_transform(cfg.model),
            merge_mode='max'
        )
    else:
        model = factory.get_model(cfg.model)
    device = factory.get_device(args.gpu)
    model.cuda()
    model.to(device)
    
    model.load_state_dict(torch.load(cfg.working_dir / 'weight' / cfg.test.weight_name))

    # get optimizer.
    print(f'optimizer: {cfg.optim.name}')
    plist = [{'params': model.parameters(), 'lr': cfg.optim.lr}]
    optimizer = factory.get_optimizer(cfg.optim)(plist)

    # get loss.
    print(f'loss: {cfg.loss.name}')
    loss = factory.get_loss(cfg.loss)

    # get scheduler.
    print(f'scheduler: {cfg.scheduler.name}')
    scheduler = factory.get_scheduler(cfg.scheduler, optimizer)

    # run model.
    runner = Runner(cfg, model, loss, optimizer, scheduler, device, my_logger)
    pred = runner.test(dataset_test, loader_test)
    
    sub = pd.read_csv(cfg.test.data_path)
    sub['target'] = np.argmax(pred, axis=1)
    sub[['target']].to_csv(cfg.working_dir / f'sub_{cfg.version}.csv', index=False)
    pd.DataFrame(pred).to_csv(cfg.working_dir / f'sub_{cfg.version}_raw.csv', index=False)
    print('submission done.')

def validation():

    args = util.get_args()
    cfg = Config.fromfile(args.config)
    cfg.fold = args.fold
    cfg.working_dir = cfg.output_dir / cfg.version / str(cfg.fold)
    print(f'version: {cfg.version}')
    print(f'fold: {cfg.fold}')

    # set logger.
    my_logger = None

    # set seed.
    util.seed_everything(cfg.seed)

    # get dataloader.
    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    dataset_test, loader_test = factory.get_dataset_loader(cfg.valid, [cfg.fold])

    ###############
    # get model.  #
    ###############

    print(f'model: {cfg.model.name}')
    if cfg.tta:
        base_model = factory.get_model(cfg.model)
        model = tta.ClassificationTTAWrapper(
            base_model,
            factory.get_tta_transform(cfg.model),
            merge_mode='max'
        )
    else:
        model = factory.get_model(cfg.model)
    device = factory.get_device(args.gpu)
    model.cuda()
    model.to(device)
    
    model.load_state_dict(torch.load(cfg.working_dir / 'weight' / cfg.test.weight_name))

    # get optimizer.
    print(f'optimizer: {cfg.optim.name}')
    plist = [{'params': model.parameters(), 'lr': cfg.optim.lr}]
    optimizer = factory.get_optimizer(cfg.optim)(plist)

    # get loss.
    print(f'loss: {cfg.loss.name}')
    loss = factory.get_loss(cfg.loss)

    # get scheduler.
    print(f'scheduler: {cfg.scheduler.name}')
    scheduler = factory.get_scheduler(cfg.scheduler, optimizer)

    # run model.
    runner = Runner(cfg, model, loss, optimizer, scheduler, device, my_logger)
    pred = runner.test(dataset_test, loader_test)
    print(pred.shape)
        
    # train_with_fold.csvからvalidのfoldを取り出す
    sub = pd.read_csv(cfg.valid.data_path)
    sub = sub[sub.fold.isin([cfg.fold])]
    sub['pred'] = np.argmax(pred, axis=1)
    
    score = loss.calc_metrics_f1(sub.pred.values, sub.target.values)
    print(f'oof metric: {score:.4f}')

    sub.to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}.csv', index=False)
    pd.DataFrame(pred).to_csv(cfg.working_dir / f'oof_{cfg.version}_{score:.4f}_raw.csv', index=False)
    print(f'oof file was saved to: oof_{cfg.version}_{score:.4f}.csv')
    
    util.line_notify(f'{cfg.version}\ntrainning done.\noof metric: {score:.4f}')

if __name__ == '__main__':
    #train()
    submission()
    validation()

