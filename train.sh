echo training_1

python train.py \
  --data_path /home/yons/depth_exp/data/ \
  --val_path /home/yons/depth_exp/data/ \
  --train_split ./splits/nyu_train_0_10_20_30_40_21483-exceptfailed-21465.txt \
  --vps_path /home/yons/depth_exp/data/nyu_p2net_vps_thresh60/ \
  --log_dir ../logs/ \
  --model_name 1 \
  --batch_size 4 \
  --num_epochs 50 \
  --start_epoch 0 \
  --using_disp2seg \
  --using_normloss \
  --load_weights_folder /home/yons/depth_exp/data/ckpts/weights_5f/ \
  --lambda_planar_reg 0.1 \
  --lambda_norm_reg 0.05 \
  --planar_thresh 200 \
