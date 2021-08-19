# StructDepth
PyTorch implementation of our ICCV2021 paper: 

StructDepth: Leveraging the structural regularities for self-supervised indoor depth estimation
![Image text](https://github.com/SJTU-ViSYS/StructDepth/blob/main/pic/show.png)
Please consider citing our paper in your publications if the project helps your research.
```
@inproceedings{structdepth,
  title={StructDepth: Leveraging the structural regularities for self-supervised indoor depth estimation},
  author={Li, Boying and Huang, Yuan and Liu, Zeyu and Zou, Danping and Yu, Wenxian},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}

```

## Getting Started

### Installation
The Python and PyTorch versions we use:

python=3.6

pytorch=1.7.1=py3.6_cuda10.1.243_cudnn7.6.3_0

**Step1**: Creating a virtual environment

```bash
conda create -n struct_depth python=3.6
conda activate struct_depth
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

**Step2**: Download [the modified scikit_image package](https://drive.google.com/file/d/1RYOwfdzM6keM3-pkWdydYJjBNSrL6gTJ/view?usp=sharing) , in which the input parameters of the Felzenswalb algorithm have been changed to accommodate our method.

```bash
unzip scikit-image-0.17.2.zip
cd scikit-image-0.17.2
python setup.py build_ext -i
pip install -e .
``` 

**Step3**: Installing other packages

```bash
pip install -r requirements.txt
```

### Download pretrained model
Please download [pretrained models](https://drive.google.com/drive/folders/1G7FLYEzhmXTZED7kKepLwYEd9a6HLT47?usp=sharing) and unzip them to MODEL_PATH

### Inference single image
```python
python inference_single_image.py --image_path=/path/to/image --load_weights_folder=MODEL_PATH
```

## Evaluation

### Download test dataset
Please download [test dataset](https://drive.google.com/drive/folders/1rJdV6j-1QF40n6Lqcn54mKnXblmSAa9q?usp=sharing)

It is recommended to unpack all test data and training data into the same data path and then modify the DATA_PATH when running a training or evaluation script.

### Evaluate NYUv2/InteriorNet/ScanNet depth or norm
Modify the evaluation script in eval.sh to evaluate NYUv2/InteriorNet/ScanNet depth and norm separately
```bash
python evaluation/nyuv2_eval_norm.py \
  --data_path DATA_PATH \
  --load_weights_folder MODEL_PATH \
```

## Trainning

### Download NYU V2 dataset
The raw NYU dataset is about 400G and has 590 videos. You can download the raw datasets from [there](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip)

### Extract Main directions
```python
python extract_vps_nyu.py --data_path DATA_PATH --output_dir VPS_PATH --failed_list TMP_LIST -- thresh 60 
```
If you need to train with a random flip, run the main direction extraction script on the images before and after the flip(add --flip) in advance, and note the failure examples, which can be skipped by referring to the code in datasets/nyu_datases.py.

### Training
Modify the training script train.sh for PATH or different trainning settings.
```bash
python train.py \
  --data_path DATA_PATH \
  --val_path DATA_PATH \
  --train_split ./splits/nyu_train_0_10_20_30_40_21483-exceptfailed-21465.txt \
  --vps_path VPS_PATH \
  --log_dir LOG_PATH \
  --model_name 1 \
  --batch_size 32 \
  --num_epochs 50 \
  --start_epoch 0 \
  --using_disp2seg \
  --using_normloss \
  --load_weights_folder PRETRAIN_MODEL_PATH \
  --lambda_planar_reg 0.1 \
  --lambda_norm_reg 0.05 \
  --planar_thresh 200 \
```
## Acknowledgement
We borrowed a lot of codes from [scikit-image](https://github.com/scikit-image/scikit-image), [monodepth2](https://github.com/nianticlabs/monodepth2), [p2net](https://github.com/svip-lab/Indoor-SfMLearner), and [LEGO](https://github.com/zhenheny/LEGO). Thanks for their excellent works!
