# TransOAR - A 3D medical Detection Transformer library 

The TransOAR project was initially developed for Transformer-based organs-at-risk detection and contains code of four 3D detection algorithms, namely DETR, Deformable DETR, RetinaNet / Retina U-Net, and Focused Decoder. To experiment with the provided arcitectures, please refer to the 'Featured detectors' section.

## Focused Decoder
Focused Decoder leverages information from an anatomical region atlas to simultaneously deploy query anchors and restrict the cross-attention’s field of view to regions of interest, which allows for a precise focus on relevant anatomical structures.


## SwinFPN[[5]](#5)
To include 3D Swin Transformer blocks in the FPN backbone, please activate the flag `use_encoder_attn` in the respective config files.\
We additionally experimented with 3D Deformable DETR encoder blocks as additional refinement stages after the FPN backbone. To activate these 3D Deformable DETR encoder blocks set `use_encoder_attn` to `True`.

## Featured detectors
It should be mentioned that the general structure of the detection pipeline remains the same for all featured detectors. To this end, all detectors use the same training pipeline and same FPN backbone.\
To access the featured detectors please checkout the linked branchs.

[Focused Decoder](https://github.com/bwittmann/transoar): A novel medical Detection Transformer restricting cross-attention’s field of view.

[DETR](https://github.com/bwittmann/transoar/tree/attn-fpn-detr)[[1]](#1): A 3D implementation of the original Detection Transformer DETR.

[Def DETR](https://github.com/bwittmann/transoar/tree/attn-fpn-def-detr)[[2]](#2): A 3D implementation of Deformable DETR.

[RetinaNet](https://github.com/bwittmann/transoar/tree/retina-unet)[[3]](#3)[[4]](#4): Adapted from the cited sources to fit our training pipeline.

## Getting started

### Installation
`pip install -e .`

### Datasets
We provide preprocessing scripts for two datasets. To prepare the datasets follow the instructions bellow:

#### VISCERAL anatomy benchmark[[6]](#6):
1) Download the CT images contained in the [Gold Corpus](https://visceral.eu/benchmarks/anatomy3-open/) (SC) and [Silver Corpus](https://visceral.eu/news/new-article-page-35/) (GC) subsets. The structure of the SC and GC subsets should be as follows:
```
GC/SC subset/
└── case_id (eg. 10000005_1)/
    └──case_id_CT_wb.nii.gz
    └──case_id_CT_wb_seg.nii.gz
```
2) Crop the images (whole body and thorax and abdomen) to foreground region presented by the annotated structures.
3) Update paths to SC and GC subsets in `./config/preprocessing_visceral.yaml`.
4) Run `python prepare_dataset_visceral.py` to generate the preprocessed dataset stored under `./datasets`.

#### AMOSS22 challenge[[7]](#7):
1) Download the training data of the [first stage](https://amos22.grand-challenge.org/). The structure should be as follows:
```
AMOS22/
└── imagesTr/
    └── amos_case_id.nii.gz
└── imagesTs/
    └── amos_case_id.nii.gz
└── labelsTr/
    └── amos_case_id.nii.gz
└── task1_dataset.json
└── task2_dataset.json
```
3) Update paths to data of the challenge's first stage in `./config/preprocessing_amos.yaml`.
4) Run `python prepare_dataset_amos.py` to generate the preprocessed dataset stored under `./datasets`.


### Training:
To train Focused Decoder on the two datasets run:
```python ./scripts/train.py --config attn_fpn_foc_dec_amos.yaml```
```python ./scripts/train.py --config attn_fpn_foc_dec_visceral.yaml```

### Testing:
To evaluate Focused Decoder's performance on the test sets run:
```python ./scripts/test.py --run <name of checkpoint in folder runs> --full_labeled```

### Checkpoints:
coming soon.

## References
<a id="1">[1]</a> 
Carion et al., "End-to-end object detection with transformers", EVVC, 2020, https://github.com/facebookresearch/detr.

<a id="2">[2]</a> 
Zhu et al., "Deformable DETR: Deformable transformers for end-to-end object detection", ICLR, 2021, https://github.com/fundamentalvision/Deformable-DETR.

<a id="3">[3]</a> 
Baumgartner et al., "nnDetection: A self-configuring method for medical object detection", MICCAI, 2021, https://github.com/MIC-DKFZ/nnDetection.

<a id="4">[4]</a> 
Jaeger, et al., "Retina U-Net: Embarrassingly simple exploitation of segmentation supervision for medical object detection", PMLR ML4H, 2020, https://github.com/MIC-DKFZ/medicaldetectiontoolkit.

<a id="5">[5]</a> 
Wittmann, et al., "SwinFPN: Leveraging Vision Transformers for 3D Organs-At-Risk Detection", MIDL, 2022.

<a id="6">[6]</a> 
Jimenez-del Toro, et al., "Cloud-based evaluation of anatomical structure segmentation and landmark detection algorithms: VISCERAL anatomy benchmarks", IEEE TMI, 2016, https://visceral.eu/benchmarks.

<a id="7">[7]</a> 
AMOS 2022: Multi-Modality Abdominal Multi-Organ Segmentation Challenge 2022, MICCAI, 2022, https://amos22.grand-challenge.org/.



