## GRIT: Faster and Better Image captioning Transformer (ECCV 2022)

This is the code implementation for the paper titled: "GRIT: Faster and Better Image-captioning Transformer Using Dual Visual Features" (Accepted to ECCV 2022) [[Arxiv](https://arxiv.org/abs/2207.09666)].


## Introduction

This paper proposes a Transformer neural architecture, dubbed <b>GRIT</b> (Grid- and Region-based Image captioning Transformer), that effectively utilizes the two visual features to generate better captions. GRIT replaces the CNN-based detector employed in previous methods with a DETR-based one, making it computationally faster.


<div align=center>  
<img src='.github/grit.png' width="100%">
</div>


## Model Zoo
| Model                                           | Task             | Checkpoint                                                                                           |
|-------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------|
| Pretrained object detector (B) on 4 OD datasets | Object Detection | [GG Drive link](https://drive.google.com/file/d/1xERJN3CvQcUcwgRZd31CUsnep_xnELcs/view?usp=share_link)  |

## Installation

### Requirements
* Python >= 3.9, CUDA >= 11.3
* PyTorch >= 1.12.0, torchvision >= 0.6.1
* Other packages: pycocotools, tensorboard, tqdm, h5py, nltk, einops, hydra, spacy, and timm

* First, clone the repository locally:
```shell
git clone https://github.com/myh4832/Final_Project.git
cd grit
```
* Then, create an environment and install PyTorch and torchvision:
```shell
conda create -n grit python=3.9
conda activate grit
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# ^ if the CUDA version is not compatible with your system; visit pytorch.org for compatible matches.
```
* Install other requirements:
```shell
pip install -r requirements.txt
python -m spacy download en
```
* Install Deformable Attention:
```shell
cd models/ops/
python setup.py build develop
python test.py
```

## Usage


### Data preparation

Download and extract COCO 2014 for image captioning including train, val, and test images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco_caption/
├── annotations/  # annotation json files and Karapthy files
├── train2014/    # train images
├── val2014/      # val images
└── test2014/     # test images
```
* Copy the files in `data/` to the above `annotations` folder. It includes `vocab.json` and some files containing Karapthy ids.

### Training

The model is trained with default settings in the configurations file in `configs/caption/coco_config.yaml`:
The training process takes around 16 hours on a machine with 8 A100 GPU.
We also provide the code for extracting pretrained features (freezed object detector), which will speed up the training significantly.

* With default configurations (e.g., 'parallel Attention', pretrained detectors on VG or 4DS, etc):
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path