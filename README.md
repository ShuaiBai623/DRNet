## DRNet

:trophy:News: **DRNet get the best EAO on the public benchmark dataset of the VOT2019 challenge**

### Introduction
The proposed tracker named DRNet, which consists of target-location network and scale regression network. They share the same backbone feature extraction network. The target-location network use a discriminative correlation filters (DCF) module to regress the confidence map. Then the response peak position of the confidence map and the target size in the previous frame is selected as the size of a dynamic anchor. The siamese scale regression network extract search and template features according to the dynamic anchor and  regress the offsets to predict the final box. In particular, we introduce a novel loss called Iterative Reweighted (IR) loss to online update the DCF module and offline train the scale regression network. Specially, the best result uses the backbone feature (ResNet50 and SE_ResNet50). We use the [pytracking](https://github.com/visionml/pytracking/) as our framework.

### Dependencies
- python 3.7.3
- pytorch (0.4.1)
- opencv (4.1.0.24)
- [pytracking](https://github.com/visionml/pytracking/tree/pytorch041)
- torchvison (0.2.2)
- Cuda


### Preparation
1. Get the pretrained models from [here](https://drive.google.com/drive/folders/12Ux4B_bJXfMtuGFQ0tAzEwnHr7lQLqPa?usp=sharing), which consists of `Res50.pth` and  `SE_Res50.pth`. Please put the models to `DRNet/pytracking/networks/`.
2. Install dependencies. 
`bash install.sh conda_install_path pytracking`

Note: We follow the installation instructions from [pytracking@pytorch041](https://github.com/visionml/pytracking/tree/pytorch041). Detailed installation instructions refers to [here](https://github.com/visionml/pytracking/blob/pytorch041/INSTALL.md).

### Usage
1. `conda activate pytracking`
2. `cd pytracking`
3. `export PYTHONPATH=$code_path:$PYTHONPATH`
4. set the absolute `CODE_PATH` and  `TRAX_BUILD_PARH`  in `DRNet/pytracking/VOT2019/tracker_DRNet.m`, the example is `DRNet/pytracking/VOT2019/tracker_DRNet_example.m`.

5. run the vot-toolkit.



