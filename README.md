# nishika_cable_6th

# Nishika_cable

Identification of cable connector type

https://www.nishika.com/competitions/19/summary

## submit_model

Softvoting　model_010 and model_044

|  exp  | architecture  |   cv   | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|-------|---------------|--------|--------|--------|--------|--------|--------|
|  010  | vit_b_16_224  | 0.9868 | 0.9938 | 0.9801 | 0.9938 | 0.9863 | 0.9801 |
|  044  | vit_b_16_224k | 0.9876 | 0.9913 | 0.9838 | 0.9950 | 0.9838 | 0.9838 |

## 条件

|  exp  | architecture  | fold 4 | diff                               |
|-------|---------------|--------|------------------------------------|
|  020  | resnet18d     | 0.9539 | Original                           |
|  021  | resnet18d     | 0.9589 | img_size 224 >> 384                |
|  022  | resnet18d     | 0.9477 | Btsize 20 >> 32                    |
|  023  | resnet18d     | 0.9502 | Btsize 32 >> 20 & Dset aff >> ori  |
|  024  | resnet18d     | 0.9215 | albumen normalize(ImageNet)        |
|  025  | resnet18d     | 0.9664 | lr 1e-4, normalize(all 0.5)        |
|  026  | resnet18d     | 0.9651 | remove randbrightness              |
|  027  | resnet18d     | 0.9676 | add transpose, remove blur, bright |
|  028  | resnet18d     | 0.9601 | optimizer sam, 10Epoch             |
|  029  | resnet18d     | 0.9751 | SAM 50 Epoch                       |
|  030  | resnet18d     | 0.9776 | Add TTA Flip, size 384 >>224       |
|  031  | resnet18d     | 0.9701 | Off mix-cut up during last 5 epoch |
|  032  | resnet18d     | 0.9726 | mixup alpha 0.4 -> 0.2             |
|  033  | resnet18d     | 0.9689 | cutmix alpha 0.4 >> 1.0            |
|  034  | vit_b_16_244  | 0.9801 | change architecture                |

## 0828

様々なネットワークで実験をすすめた
シングルモデルで最も良いのは010、画像サイズとバッチサイズを大きくすると結果は悪化した
exp013よりttaの条件を追加
014よりHorizontalFlipは削除、形が左右非対称なケーブルの規格があるため

## 1010

学習と評価のデータセットは2classのCenterNetでとりだすことにした
スコアが低い、正方形で切り出せないものはそのままのサイズでphotos3に保存
fold4を対象に条件の最適化を探索する
コードのリファクタリング
混合行列をみてると特定のクラスの落ち込みがUSB micro BWの精度が悪いことが判明
そのクラスのみを判別する2classだと精度はどこまで良いのか調べる。

