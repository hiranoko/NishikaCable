import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import f1_score, accuracy_score
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

args = get_args()

version = args.version
folds = [0,1,2,3,4]

sub = pd.read_csv('../input/test_with_fold.csv')
pred = np.zeros((len(sub), 15))

train = pd.read_csv('../input/train_with_fold.csv')
target = train.target.values
oof = pd.DataFrame()
oof_cls = pd.DataFrame()

for fold in folds:
    df = pd.read_csv(f'../output/{version}/{fold}/sub_{version}_raw.csv')
    pred += df.values
    oof_raw_name = [x for x in os.listdir(f'../output/{version}/{fold}/') if ('oof' in x)&('raw' in x)]
    oof_cls_name = [x for x in os.listdir(f'../output/{version}/{fold}/') if ('oof' in x)&('raw' not in x)]

    print(oof_raw_name)
    print(oof_cls_name)

    _oof = pd.read_csv(f'../output/{version}/{fold}/{oof_raw_name[0]}')
    oof = pd.concat([oof, _oof], axis=0)

    _oof = pd.read_csv(f'../output/{version}/{fold}/{oof_cls_name[0]}')
    oof_cls = pd.concat([oof_cls, _oof], axis=0)

oof.columns = ['conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4','conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9','conf_10', 'conf_11', 'conf_12', 'conf_13', 'conf_14']

_oof = oof.copy()

_oof['pred'] = np.argmax(_oof.values, axis=1)

oof_pred = _oof.pred.values
oof_target = oof_cls.target

assert len(oof)==len(train)
score = f1_score(oof_target, oof_pred, average='micro')
print(score)

oof_cls.pred = oof_pred
oof_cls = pd.concat([oof_cls, oof], axis=1)

pred /= len(folds)

sub.target = np.argmax(pred, axis=1)

sub[['conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4','conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9','conf_10', 'conf_11', 'conf_12', 'conf_13', 'conf_14']] = pred
sub[['target','conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4','conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9','conf_10', 'conf_11', 'conf_12', 'conf_13', 'conf_14']].to_csv(f'../output/sub_{version}.csv', index=False)
oof_cls.to_csv(f'../output/results/oof_{version}_{score:.4f}.csv', index=False)
