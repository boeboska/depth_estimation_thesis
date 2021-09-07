# Transformer-based guidance off Self-Supervised depth estimation

In this repository [the results (Thesis_depth_estimation___Bob_Borsboom___10802975.pdf)](Thesis_depth_estimation___Bob_Borsboom___10802975.pdf) and code are published regarding the master thesis of Bob Borsboom for the Master Artificial Intelligence at the University of Amsterdam.


<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>
This visualization comes from the original monodepth paper [https://github.com/nianticlabs/monodepth2]




## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```

## 🖼️ Prediction for a single image

You can predict depth for a single image on a pretrained model with:
```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

On its first run this will download the `mono+stereo_640x192` pretrained model (99MB) into the `models/` folder.
We provide the following  options for `--model_name`:


| `--model_name`          | Training modality | Imagenet pretrained? | Model resolution  | KITTI abs. rel. error |  delta < 1.25  |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|
| [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip)          | Mono              | Yes | 640 x 192                | 0.115                 | 0.877          |



## 💾 KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!




## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

```shell
python train.py --model_name mono_model --png
```
To reproduce the experiment results the following settings must be set:



  <strong>Edge loss experiment: </strong>
  <ul>
  <li> Run 'python train.py --png --edge_loss true --edge_detection_threshold 0.1 --edge_weight 2e-4  </li>
  </ul>
  
  <strong>Weight mask experiment:</strong>
  <ul>
  <li> Run 'python train.py --png --attention_mask_loss true --attention_mask_threshold 1.05 --attention_weight 5 --reduce_attention_weight 0.9 </li>
  </ul>
  <strong> Transformer block experiment: </strong>
  <ul>
  <li> Run 'python train.py --png --self_attention true </li>
  </ul>

For other training options, see options.py such as learning rate and ablation settings.



## 📊 KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono
```


