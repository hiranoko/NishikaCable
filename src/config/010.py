from pathlib import Path
import os

version = f'010'
seed = 1111

n_fold = 5
n_class = 15
num_workers = 4
target_size = (224,224) # None=Original size.
use_amp = True
use_mixup_cutmix = True
num_gpu = 1
batch_size = 20*num_gpu
mixup_alpha = 0.4
cutmix_alpha = 0.4
mixup_freq = 0.8
tta = False
sam_optimizer = False

wo_mixup_epochs = 501
n_epochs = 50

project = 'Nishika_cable'
input_dir = Path(f'/home/hirano/work/Competition/{project}/input')
output_dir = Path(f'/home/hirano/work/Competition/{project}/output')


# dataset
dataset_name = 'CustomDatasetClassificationAffine'

# model config
# model
model = dict(
    name = 'NishikaCustomModelClass',
    architecture = 'vit_base_patch16_224',
    pretrained_weight=True,
    params = dict(
    )
)

# optimizer
optim = dict(
    name = 'AdamW',
    lr = 1e-5*num_gpu,
    weight_decay = 0.01
)

# loss
loss = dict(
    name = 'CustomLossClassification',
    params = dict(
    ),
)

# scheduler
scheduler = dict(
    name = 'CosineAnnealingLR',
    params = dict(
        T_max=n_epochs,
        eta_min=0,
        last_epoch=-1,
    )
)


# snapshot
snapshot = dict(
    save_best_only=True,
    mode='max',
    initial_metric=None,
    name=version,
    monitor='metric'
)

# logger
logger = dict(
    params=dict(
        logging_info=['loss', 'metric'],
        print=False
    ),
)

# augmentations.
horizontalflip = dict(
    name = 'HorizontalFlip',
    params = dict()
)

verticalflip = dict(
    name = 'VerticalFlip',
    params = dict()
)

shiftscalerotate = dict(
    name = 'ShiftScaleRotate',
    params = dict(
        shift_limit = 0.1,
        scale_limit = 0.1,
        rotate_limit = 15,
    ),
)

gaussnoise = dict(
    name = 'GaussNoise',
    params = dict(
        var_limit = 5./255.
        ),
)

blur = dict(
    name = 'Blur',
    params = dict(
        blur_limit = 3
    ),
)

randommorph = dict(
    name = 'RandomMorph',
    params = dict(
        size = target_size,
        num_channels = 1,
    ),
)

randombrightnesscontrast = dict(
    name = 'RandomBrightnessContrast',
    params = dict(),
)

griddistortion = dict(
    name = 'GridDistortion',
    params = dict(),
)

elastictransform = dict(
    name = 'ElasticTransform',
    params = dict(
        sigma = 50,
        alpha = 1,
        alpha_affine = 10
    ),
)

cutout = dict(
    name = 'Cutout',
    params = dict(
        num_holes=1,
        max_h_size=int(256*0.3),
        max_w_size=int(256*0.3),
        fill_value=0,
        p=0.7
    ),
)

totensor = dict(
    name = 'ToTensorV2',
    params = dict(),
)

oneof = dict(
    name='OneOf',
    params = dict(),
)

normalize = dict(
    name = 'Normalize',
    params = dict(),
)


# train.
train = dict(
    is_valid = False,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [
        horizontalflip,
        shiftscalerotate,
        blur,
        randombrightnesscontrast,
        totensor
        ],
)


# valid.
valid = dict(
    is_valid = True,
    data_path = input_dir / f'train_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [totensor],
)

# test.
test = dict(
    is_valid = True,
    data_path = input_dir / 'test_with_fold.csv',
    img_dir = input_dir / 'photos',
    target_size = target_size,
    dataset_name = dataset_name,
    normalize = normalize,
    weight_name = f'{version}_best.pt',
    loader=dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
    transforms = [totensor],
)