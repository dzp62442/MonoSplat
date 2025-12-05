在 OmniScene 数据集上与 SVF-GS 进行对比

### 训练
```bash
python -m src.main +experiment=omniscene_112x200 \
mode=train \
output_dir=checkpoints/omniscene-112x200 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth
```

### 测试

> 默认使用 mini-test 模式，如需完整测试，请手动注释 `dataset_omniscene.py` 中的抽样语句。

```bash
python -m src.main +experiment=omniscene_112x200 \
mode=test \
checkpointing.load=checkpoints/omniscene-112x200/checkpoints/epoch_0-step_100000.ckpt \
test.output_path=outputs/omniscene-112x200 \
wandb.mode=disabled \
test.compute_scores=true
```

- 保存可视化结果
```bash
test.save_image=true \
test.save_video=true \
test.save_video_omniscene=true
```
- 未实现的可视化：`save_gt_image`、`save_depth`、`save_gaussian` 等。

---

<p align="center">
  <h1 align="center">MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models</h1>


## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

```bash
git clone https://github.com/CUHK-AIM-Group/MonoSplat.git
cd MonoSplat
conda create -n monosplat python=3.10
conda activate monosplat
pip install numpy==1.26.3 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

Our MonoSplat uses the same training datasets as pixelSplat and MVSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

### DTU (For Testing Only)

* Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).
* Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`

## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155195605_link_cuhk_edu_hk/EepgRlcdKLpOqSjjQNejr3QBy9BE566ljNBvs3C9ck1Jyg?e=YSIEgb), and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=checkpoints/epoch_63-step_300000.ckpt \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    test.compute_scores=true \
    test.save_image=false \
    wandb.mode=disabled
```

* the rendered novel views will be stored under `outputs/test`

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimatch and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

Our models are trained with a single A100 (80GB) GPU. They can also be trained on multiple GPUs with smaller RAM by setting a smaller `data_loader.train.batch_size` per GPU.

### Cross-Dataset Generalization

We employ our baseline model, which was trained using the RealEstate10K dataset, to perform extensive cross-dataset validation. For instance, to execute an evaluation on the DTU benchmark, simply issue the following command in the terminal:


```bash
# RealEstate10K -> DTU
python -m src.main +experiment=dtu \
    mode=test \
    checkpointing.load=/path/to/checkpoint \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled
```

## BibTeX

```bibtex
@article{liu2025monosplat,
  title={MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models},
  author={Liu, Yifan and Fan, Keyu and Yu, Weihao and Li, Chenxin and Lu, Hao and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2505.15185},
  year={2025}
}
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat ). Many thanks to these two projects for their excellent contributions!
