# StylerDALLE
Official PyTorch implementation for ICCV 2023 paper [StylerDALLE: Language-Guided Style Transfer Using a Vector-Quantized Tokenizer of a Large-Scale Generative Model](https://arxiv.org/pdf/2303.09268.pdf).

![](images/teaser.png)

## Updates
_17 Jul 2023_: StylerDALLE is accepted by ICCV 2023.

_16 Aug 2023_: StylerDALLE is also accepted for a presentation at the [MMFM Workshop](https://sites.google.com/view/iccv-mmfm/home?authuser=0) @ICCV, see you in Paris.

_13 Sep 2023_: Training code for _StylerDALLE-1_ and _StylerDALLE-Ru_  released.

## Setup:
Environment:
```
conda env create -f environment.yml
conda activate stylerdalle
```

## Usage:

### Data Pre-Processing

Before training, we preprocess the [COCO](https://cocodataset.org/#home) dataset. 
Specifically, we encode the images into discrete tokens with a pretrained vector-quantized tokenizer.
First of all, you shall download the images ([train-2014](http://images.cocodataset.org/zips/train2014.zip), [val-2014](http://images.cocodataset.org/zips/val2014.zip)) and the [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip).

Preprocess for StylerDALLE-1 with the officially released dVAE of [DALL-E](https://github.com/openai/DALL-E):
```python prep/prep_coco.py ```

Preprocess for StylerDALLE-Ru with the VQGAN of [Ru-DALLE](https://github.com/ai-forever/ru-dalle):
```python prep/prep_coco_ru.py ```

In addition, for reinforcement learning, we prepare the caption data files [here](https://drive.google.com/drive/folders/1V14G1ddKKl7PbbLOwfZ_jDr8HttvVdYA?usp=sharing).
They are derived from the original coco annotations but contain only successfully preprocessed image data.

### Train StylerDALLE-1
- ***Self-supervised Pre-training***:
    
    ```
    python train.py
    ```

- ***Reinforcement Learning*** :
    ```
    python train_rl.py --styl 'a Van Gogh style oil painting'
    ```

### Train StylerDALLE-Ru
- ***Self-supervised Pre-training***:
    
    ```
    python train_ru.py
    ```

- ***Reinforcement Learning*** :
    ```
    python train_ru_rl.py --styl 'a Van Gogh style oil painting'
    ```
    
## Reference
```
@InProceedings{Xu2023StylerDALLE,
    author    = {Xu, Zipeng and Sangineto, Enver and Sebe, Nicu},
    title     = {StylerDALLE: Language-Guided Style Transfer Using a Vector-Quantized Tokenizer of a Large-Scale Generative Model},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7601-7611}
}
```
