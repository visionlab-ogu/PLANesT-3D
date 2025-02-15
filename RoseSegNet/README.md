## RoseSegNet
![](https://github.com/visionlab-ogu/PLANesT-3D/blob/main/RoseSegNet/teaser_rosesegnet.png)


## Overview
The network processes regions of points in a hierarchical manner, where at each level, point features are aggregated using attention-based operators.

## Citation
If you find our research useful, please consider citing:

```
@article{turgut2022,
    title  = {RoseSegNet: An attention-based deep learning architecture for
              organ segmentation of plants},
    journal = {Biosystems Engineering},
    volume  = {221},
    pages   = {138-153},
    year    = {2022},
    issn    = {1537-5110},
    doi     = {https://doi.org/10.1016/j.biosystemseng.2022.06.016},
    author  = {Kaya Turgut and Helin Dutagaci and David Rousseau}
}
```


## Installation
Thanks to the [PointNet++](https://github.com/charlesq34/pointnet2) architecture for sharing their repository publicly. This architecture builds upon the original PointNet++ structure. You can follow the installation instructions from PointNet++ to set up the environment dependencies and compile the TensorFlow operator.

## Usage
<strong>Data Preparation:</strong>
The data preparation process involves three steps: 1) Sampling the point cloud and extracting eigenvalues for various radii (which is not used in this work), 2) Splitting the data into training and test samples and 3) Dividing each point clouds to pillars by following [PointCNN](https://github.com/yangyanli/PointCNN).

```
python tools/get_sampling_eigen_values.py
python tools/split_train_test.py
python tools/blocks/prepare_seg_data.py
```

<strong>Train:</strong> To begin the training process for segmentation, run the following command:
```
python train.py
```

<strong>Evaluation:</strong> After training, to obtain predictions for each point in the point cloud and calculate the metrics, run the following commands:
```
python test.py
python eval.py
```



