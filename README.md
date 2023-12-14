## GRIT: Faster and Better Image captioning Transformer (ECCV 2022)

<div align=center>  
<img src='.github/grit.png' width="100%">
</div>


## Pretrained Object Detector
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
├── annotations/  # annotation json files
├── train2014/    # train images
├── val2014/      # val images
```

### Training

The model is trained with default settings in the configurations file in `configs/caption/coco_config.yaml`:

* With default configurations (e.g., 'parallel Attention', pretrained detectors on 4DS, etc):
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=./detector_checkpoint_4ds.pth
```

### Final Print Result

최종 결과는 grit/print_result.py를 실행하면 출력됩니다.

```shell
python print_result.py
```