# Google Landmark Recogntion and Retrieval 2021 Solutions

In this repository you can find solution and code for [Google Landmark Recognition 2021](https://www.kaggle.com/c/landmark-recognition-2021) and [Google Landmark Retrieval 2021](https://www.kaggle.com/c/landmark-retrieval-2021) competitions (both in top-100).

## Brief Summary 
My solution is based on the latest modeling from the previous competition and strong post-processing based on re-ranking and using side models like detectors. **I used single RTX 3080, EfficientNet B0 and only competition data for training.**

##  Model and loss function

I used the same model and loss as the winner team of the previous competition as a base. Since I had only single RTX 3080, I hadn't enough time to experiment with that and change it. The only things I managed to test is Subcenter ArcMarginProduct as the last block of model and ArcFaceLossAdaptiveMargin loss function, which has been used by the 2nd place team in the previous year. Both those things gave me a signifact score boost (around 4% on CV and 5% on LB). 

## Setting up the training and validation

### Optimizing and scheduling
Optimizer - Ranger (lr=0.003)  
Scheduler - CosineAnnealingLR (T_max=12) + 1 epoch Warm-Up

### Training stages
I found the best perfomance in training for 15 epochs and 5 stages:  
I (1-3) - Resize to image size, Horizontal Flip  
II (4-6) - Resize to bigger image size, Random Crop to image size, Horizontal Flip  
III (7-9) - Resize to bigger image size, Random Crop to image size, Horizontal Flip, Coarse Dropout with one big square (CutMix)  
IV (10-12) - Resize to bigger image size, Random Crop to image size, Horizontal Flip, FMix, CutMix, MixUp  
V (13-15) - Resize to bigger image size, Random Crop to image size, Horizontal Flip  

I used default Normalization on all the epochs.

### Validation scheme

Since I hadn't enough hardware, this became my first competition where I wasn't able to use a K-fold validation, but at least I saw stable CV and CV/LB correlation at the previous competitions, so I used simple stratified train-test split in 0.8, 0.2 ratio.

## Inference and Post-Processing:


1. Change class to non-landmark if it was predicted more than 20 times .
2. Using pretrained YoloV5 for detecting non-landmark images. `All classes` are used, boxes with `confidence < 0.5` are dropped. If total area of boxes is greater than `total_image_area / 2.7`, the sample is marked as non-landmark. 
I tried to use YoloV5 for cleaning the train dataset as well, but it only decreased a score.
3. Tuned post-processing from [this](https://arxiv.org/abs/2010.01650) paper, based on the cosine similarity between train and test images to non-landmark ones.
4. Higher image size for extracting embeddings on inference.
5. Also using public train dataset as an external data for extracting embeddings.

## Didn't work for me
- Knowledge Distillation 
- Resnet architectures (on average they were worse than effnets)
- Adding an external non-landmark class to training from 2019 test dataset
- Train binary non-landmark classifier

Transfer Learning on the full dataset and Label Smoothing should be useful here, but I didn't have time to test it.
