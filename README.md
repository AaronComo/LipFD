# Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-syncing DeepFakes

**IJCAI 2024 Anonymous Submission #1522**

This repository contains the codes of "Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-syncing DeepFakes".

![headline](README.assets/headline.png)

> **Abstract.** In recent years, deepfake technology has achieved unprecedented success in high-quality video synthesis, whereas these methods also pose potential and severe security threats to humanity. Deepfake can be bifurcated into entertainment applications like face swapping and illicit uses such as lipsync fraud. However, lip-forgery videos, which neither change identity nor have discernible visual artifacts, present a formidable challenge to existing deepfake detection methods as they are unable to capture these subtle nuances. Experimental evidence has shown that the effectiveness of current methods often drastically decreases or even fails altogether when faced with lip-syncing videos. 
>
> In this paper, we, for the first time, propose an approach for lip-forgery detection that exploits the inconsistency between lip movements and audio signals, while also considering the correlation with head motions. We curate a high-quality audio-visual dataset as a benchmark for lipsync detection research. Based on it and other publicly available deepfake datasets, we conducted comprehensive experiments on forgery detection. Our approach achieved a 95.27% accuracy, surpassing the latest detection method by 3.49%. Additionally, the results show that our approach demonstrates the state-of-the-art performance of generalization to unseen data and robustness in detecting lip forgery videos on various perturbations. 

![pipeline](README.assets/pipeline.png)



## AVLip: A high-quality audio-visual dataset for lipsync detection

The current deepfake detection methods have reached a highly advanced level with almost 100% accuracy. However, when facing state-of-the-art lip forgery videos, their performances drop drastically. One of the most important reasons is that those lip-forged videos are omitted during training, which gives rise to the need to build a dataset that can be used to detect lipsync. Unfortunately, almost all present public DeepFake datasets only contain video or image sources and there is no dedicated dataset specifically for lip forgery available as well. To fill this gap, we propose a high-quality audio-visual dataset, AVLips, which contains a large amount of videos generated by several state-of-the-art lipsync methods.

We will open-source upon publication soon.

<div align=center><img src="README.assets/dataset.png" width="300"></div>



## Requirements

- Python 3.10.13

~~~bash
conda create -n LipFD python==3.10
conda activate LipFD
~~~

- Python packages

~~~bash
pip install -r requirements.txt
~~~



## Preprocess

Download AVLip dataset and put it in the root directory. 

AVLip dataset folder structure.

~~~bash
AVLip
├── 0_real
│   ├── 0.mp4
│    ...
├── 1_fake
│   ├── 0.mp4
│   └── ...
└── wav
    ├── 0_real
    │   ├── 0.wav
    │   └── ...
    └── 1_fake
        ├── 0.wav
        └── ...
~~~

Preprocess the dataset for training or validation.

~~~bash
python preprocess.py
~~~

It will create two folders under `datasets/`.

Preprocessed AVLip dataset folder structure.

~~~bash
datasets
├── train
│   ├── 0_real
│   │   ├── 0_0.png
│   │   └── ...
│   └── 1_fake
│       ├── 0_0.png
│       └── ...
└── val
    ├── 0_real
    │   ├── 0_0.png
    │   └── ...
    └── 1_fake
        ├── 0_0.png
        └── ...
~~~



## Validation

Download our pertained weights and save it in to `checkpoints/ckpt.pth`.

~~~bash
python validate.py --real_list_path ./datasets/val/0_real --fake_list_path ./datasets/val/1_fake --ckpt ./checkpoints/ckpt.pth
~~~



## Train

First, edit `--fake_list_path` and `--real_list_path`  in `options/base_options.py`.

Then, run `python train.py`.

