python main.py ctdet --exp_id new_twoclass --batch_size 4 --master_batch 4 --lr 1e-4 --gpus 0 --num_workers 4 --num_epochs 30 --lr_step 15,20 --load_model ../models/centernet_hardnet85_coco_608.pth --val_interval 1

python test.py ctdet --load_model ../exp/ctdet/new_twoclass/model_best.pth --nms --test_scales 1,1.25 --flip_test --exp_id new_twoclass --not_prefetch_test
