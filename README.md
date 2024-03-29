<!-- [![DOI](https://img.shields.io/badge/arXiv-https%3A%2F%2Fdoi.org%2F10.48550%2FarXiv.2207.10774-B31B1B)](https://doi.org/10.48550/arXiv.2207.10774)
[![DOI](https://img.shields.io/badge/MIDL-https%3A%2F%2Fopenreview.net%2Fforum%3Fid=yiIz7DhgRU5-B31B1B)](https://openreview.net/forum?id=yiIz7DhgRU5) -->
<!-- [![MELBA](https://img.shields.io/badge/MELBA-Focused_Decoding_Enables_3D_Anatomical_Detection_by_Transformers-red.svg)](https://www.melba-journal.org/papers/2023:003.html)
[![MIDL](https://img.shields.io/badge/MIDL-SwinFPN%3A_Leveraging_Vision_Transformers_for_3D_Organs--At--Risk_Detection-red.svg)](https://openreview.net/forum?id=yiIz7DhgRU5) -->
[![GitHub Stars](https://img.shields.io/github/stars/bwittmann/transoar?style=social)](https://github.com/bwittmann/transoar)
![visitors](https://visitor-badge.glitch.me/badge?page_id=bwittmann.transoar)


<img src="docs/detectors.png">

The TransOAR project was initially developed for Transformer-based organs-at-risk detection and contains code of three 3D Detection Transformers, namely Focused Decoder, DETR, and Deformable DETR. Additionally, we adopted RetinaNet/Retina U-Net from nnDetection into our training pipeline to ensure comparability of results with traditional CNN-based detectors.

To access the featured detectors and their detailed configs, please checkout the linked branches:

[Focused Decoder](https://github.com/bwittmann/transoar): A novel medical Detection Transformer restricting the cross-attention’s field of view.\
[DETR](https://github.com/bwittmann/transoar/tree/attn-fpn-detr) [[1]](#1): A 3D implementation of the original Detection Transformer DETR.\
[Deformable DETR](https://github.com/bwittmann/transoar/tree/attn-fpn-def-detr) [[2]](#2): A 3D implementation of Deformable DETR.\
[RetinaNet/Retina U-Net](https://github.com/bwittmann/transoar/tree/retina-unet) [[3]](#3)[[4]](#4): Adapted from the cited sources to fit our training pipeline.


# News
**February 23**: [*Focused Decoding Enables 3D Anatomical Detection by Transformers*](https://www.melba-journal.org/papers/2023:003.html) has been accepted at MELBA!\
**May 22**: [*SwinFPN: Leveraging Vision Transformers for 3D Organs-At-Risk Detection*](https://openreview.net/forum?id=yiIz7DhgRU5) has been accepted at MIDL 22!

# Focused Decoder  [![MELBA](https://img.shields.io/badge/MELBA-Focused_Decoding_Enables_3D_Anatomical_Detection_by_Transformers-red.svg)](https://www.melba-journal.org/papers/2023:003.html)
<img src="docs/foc_dec.png">\
**TL;DR**: Focused Decoder leverages information from an anatomical region atlas to simultaneously deploy query anchors and restrict the cross-attention’s field of view to RoIs, alleviating the need for large-scale annotated datasets. Focused Decoder not only delivers competitive results but also facilitates the accessibility of explainable results via attention weights.


## Usage
The usage remains the same for all branches and, therefore, all featured detectors.

### Installation
#### General
Create a new virtual environment using, for example, anaconda:

    conda create -n transoar python=3.8

and run:

    pip install -e .

The installation was tested using:
- Python 3.8
- Ubuntu 20.04
- CUDA 11.4

#### Compiling CUDA operations
To compile the CUDA operations of the deformable attention module, run:

    cd ./transoar/models/ops
    python setup.py install

Alternatively, one can experiment with the python implementation by deactivating the flag `use_cuda` in the respective config file.

To compile NMS used in RetinaNet/Retina U-Net, checkout this [branch](https://github.com/bwittmann/transoar/tree/retina-unet) and follow the general installation steps described above.


### Datasets
We provide exemplary preprocessing scripts for two publicly available datasets.
It should be mentioned that these preprocessing scripts should act as templates to experiment with additional datasets.

#### AMOSS22 challenge [[5]](#5)
1) Download the training data of the challenge's [first stage](https://amos22.grand-challenge.org/). The structure should be as follows:
```
AMOS22/
└── imagesTr/
    └── <case_id>.nii.gz
└── imagesTs/
    └── <case_id>.nii.gz
└── labelsTr/
    └── <case_id>.nii.gz
└── task1_dataset.json
└── task2_dataset.json
```
2) Update paths to the raw data in `./config/preprocessing_amos.yaml`.
3) Run `python prepare_dataset_amos.py` to generate the preprocessed dataset, which will be stored under `./dataset`.

#### VISCERAL anatomy benchmark [[6]](#6)
1) Download the CT images contained in the [Gold Corpus](https://visceral.eu/benchmarks/anatomy3-open/) (GC) and [Silver Corpus](https://visceral.eu/news/new-article-page-35/) (SC) subsets. The structure of the GC and SC subsets should be as follows:
```
GC/SC subset/
└── <case_id>/
    └── <case_id>_CT_wb.nii.gz
    └── <case_id>_CT_wb_seg.nii.gz
```
2) Update paths to GC and SC subsets in `./config/preprocessing_visceral.yaml`.
3) Run `python prepare_dataset_visceral.py` to generate the preprocessed dataset, which will be stored under `./dataset`.

### Training
First, set the `dataset` flag in the respective config file to the name of the preprocessed dataset. If necessary, modify the config file accordingly.\
To train on a specific dataset, run:
    
    python CUDA_VISIBLE_DEVICE=<gpu_id> scripts/train.py --config attn_fpn_<detector>_<dataset>.yaml

### Testing
To evaluate performances of created checkpoints on the test sets, run:

    python scripts/test.py --run <name_of_checkpoint_in_folder_runs> --num_gpu <gpu_id> --full_labeled

For visualization of results and attention maps, please check additional flags in `scripts/test.py`.


# SwinFPN [![MIDL](https://img.shields.io/badge/MIDL-SwinFPN%3A_Leveraging_Vision_Transformers_for_3D_Organs--At--Risk_Detection-red.svg)](https://openreview.net/forum?id=yiIz7DhgRU5)
This repository also contains code for SwinFPN. To include 3D Swin Transformer blocks in the FPN backbone, please activate the flag `use_encoder_attn` in the respective config files.\
We additionally experimented with 3D Deformable DETR encoder blocks as additional refinement stages after the FPN backbone. To activate these 3D Deformable DETR encoder blocks activate the flag `use_encoder_attn`.

# Citation
If you find our repository useful in your research, please consider citing::
```bibtex
@article{wittmann2023focused,
    title={Focused Decoding Enables 3D Anatomical Detection by Transformers},
    author={Wittmann, Bastian and Navarro, Fernando and Shit, Suprosanna and Menze, Bjoern},
    journal={Machine Learning for Biomedical Imaging},
    volume={2},
    issue={February 2023 issue},
    year={2023},
    pages={72--95},
    issn={2766-905X},
    url={https://melba-journal.org/2023:003}
}
@inproceedings{wittmann2022swinfpn,
    title={Swin{FPN}: Leveraging Vision Transformers for 3D Organs-At-Risk Detection},
    author={Wittmann, Bastian and Shit, Suprosanna and Navarro, Fernando and Peeken, Jan C and Combs, Stephanie E and Menze, Bjoern},
    booktitle={Medical Imaging with Deep Learning},
    year={2022},
    url={https://openreview.net/forum?id=yiIz7DhgRU5}
}
```

# References
<a id="1">[1]</a> 
Carion et al., "End-to-end object detection with transformers," EVVC, 2020, https://github.com/facebookresearch/detr.

<a id="2">[2]</a> 
Zhu et al., "Deformable DETR: Deformable transformers for end-to-end object detection," ICLR, 2021, https://github.com/fundamentalvision/Deformable-DETR.

<a id="3">[3]</a> 
Baumgartner et al., "nnDetection: A self-configuring method for medical object detection," MICCAI, 2021, https://github.com/MIC-DKFZ/nnDetection.

<a id="4">[4]</a> 
Jaeger et al., "Retina U-Net: Embarrassingly simple exploitation of segmentation supervision for medical object detection," PMLR ML4H, 2020, https://github.com/MIC-DKFZ/medicaldetectiontoolkit.

<a id="6">[5]</a> 
AMOS 2022: Multi-Modality Abdominal Multi-Organ Segmentation Challenge 2022, MICCAI, 2022, https://amos22.grand-challenge.org/.

<a id="5">[6]</a> 
Jimenez-del Toro et al., "Cloud-based evaluation of anatomical structure segmentation and landmark detection algorithms: VISCERAL anatomy benchmarks," IEEE TMI, 2016, https://visceral.eu/benchmarks.

