#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.filename.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = torch.tensor(self.target[idx]).long()
        path = str(self.path / self.name[idx])
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}

class CustomDataset_USBmicroBW(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.filename.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = torch.tensor(self.target[idx]).long()
        #if target == 12:
        #    target = torch.tensor(1).long()
        #else:
        #    target = torch.tensor(0).long()
        path = str(self.path / self.name[idx])
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}

class CustomDatasetClassification(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.filename.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = torch.tensor(self.target[idx]).long()
        path = str(self.path / self.name[idx])
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        img = cv2.resize(img, self.target_size)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}

class CustomDatasetClassificationAffine(Dataset):
    def __init__(self, cfg, folds, transforms):
        self.cfg = cfg
        self.transform = transforms

        self.df = pd.read_csv(self.cfg.data_path)
        self.df = self.df[self.df.fold.isin(folds)]

        # bellow is custom initializations.
        self.name = self.df.filename.values
        self.target = self.df.target.values
        self.path = self.cfg.img_dir
        self.target_size = self.cfg.target_size

    def resize_affine(self, _img):
        height, width = _img.shape[0], _img.shape[1]
        c = np.array([_img.shape[1] / 2., _img.shape[0] / 2.], dtype=np.float32)
        s = max(_img.shape[0], _img.shape[1])
        trans_input = self.get_affine_transform(c, s*0.9, 0, self.target_size)
        return cv2.warpAffine(_img, trans_input, self.target_size, flags=cv2.INTER_LINEAR)

    def get_affine_transform(self, center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
        
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        
        return trans

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        target = torch.tensor(self.target[idx]).long()
        path = str(self.path / self.name[idx])
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255
        
        #img = cv2.resize(img, self.target_size)
        img = self.resize_affine(img)

        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return {'image': img,
                'target': target}

