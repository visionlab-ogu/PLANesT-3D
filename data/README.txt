
PLANesT-3D Dataset
--------------------------------------------------------------


PLANesT-3D is composed of 34 point cloud models representing 34 real plants from three different plant species: Capsicum annuum, Rosa kordana, and Ribes rubrum. 

The dataset can be downloaded from the link:
https://www.dropbox.com/scl/fo/c4vqad7ak3a9pv1bkz6x1/AGxXY1yYhWzyjIGaqDwJes8?rlkey=qsukydmtgunpqe6zg8zpsjr2c&st=4l8xznqz&dl=0

There are four compressed files containing the following folders:
PLANesT-3D 
PLANesT_3D_PlantPointClouds 
PLANesT_3D_RawPointClouds
PLANesT-3D_Images

The details of the contents of these folders are provided below:



PLANesT-3D Plant Point Clouds
--------------------------------------------------------
--------------------------------------------------------

The folders "PLANesT-3D"(files in ascii) and "PLANesT_3D_PlantPointClouds"(files in binary) each contain 10 point clouds of Pepper, 10 point clouds of Rose, and 14 point clouds of Ribes plants. The point clouds were reconstructed from multiple images captured manually around each plant. For reconstruction of 3D color point clouds from 2D color images, Agisoft Metashape Professional (Agisoft LLC, St. Petersburg, Russia) was employed. The plant point cloud was separated from the background, pose normalized, and scaled to the correct size through a semi-automatic process.


In addition to the point coordinates and colors, three scalar properties are provided:
1. property : confidence
2. property : semanticLabel
3. property : instanceLabel

The confidence values are provided by the Metashape Professional. Confidence value for a point corresponds to the number of images that "observe" and contribute to reconstruct the point. 

The semanticLabel values indicate whether a point is a "stem" point or a "leaf" point. 
"stem" points have the value of SemanticLabel as "1". 
"leaf" points have the value of SemanticLabel as "2".

The instanceLabel values indicate which specific leaflet instance a point belongs to. All "stem" points have the value of InstanceLabel as "1".
"Leaf" instance IDs start from "2". The highest instance value is "N+1", where N is the number of leaflets in the plant.

The point clouds in the folder "PLANesT-3D" are in ascii format. The confidence values, semantic labels, and instance labels are provided as separate files.


The point clouds in the folder ""PLANesT_3D_PlantPointClouds" are stored in binary .ply files. The confidence values, semantic labels, and instance labels are stored within the .ply files as scalar properties.

The binary ply files can be visualized by CloudCompare (https://www.danielgm.net/cc/). CloudCompare allows the additional scalar fields to be loaded and visualized.

The python libraries plyfile and open3d can be used to read and visualize the ply files.
https://pypi.org/project/plyfile/
https://www.open3d.org/

A python script "read_visualize_ply_files.py" is provided for reading the ply files and the scalar fields.





Raw point clouds of PLANesT-3D DATASET
----------------------------------------------------------
----------------------------------------------------------

The folder "PLANesT_3D_RawPointClouds" contains raw point clouds from which the plant models of the PLANesT-3D dataset were extracted.

The point clouds were reconstructed from multiple images captured manually around each plant. For reconstruction of 3D color point clouds from 2D color images, Agisoft Metashape Professional (Agisoft LLC, St. Petersburg, Russia) was employed. The raw point clouds include all the points in the scene reconstructed by Agisoft Metashape.

The raw point clouds are stored in binary .ply files.
In addition to the point coordinates and colors, two scalar properties are provided in the ply files:
1. property : confidence
2. property : instanceLabel

The confidence values are provided by the Metashape Professional. Confidence value for a point corresponds to the number of images that "observe" and contribute to reconstruct the point. 

The instanceLabel values indicate which specific leaflet instance a point belongs to. 

All "background" points have the value of InstanceLabel as "0".
All "stem" points have the value of InstanceLabel as "1".
"Leaf" instance IDs start from "2". The highest instance value is "N+1", where N is the number of leaflets in the plant.

A python script "read_visualize_raw_point_clouds.py" is provided for reading the ply files and the scalar fields.





Images of the PLANesT-3D dataset
---------------------------------------------------------
---------------------------------------------------------

The point clouds in the PLANesT-3D dataset were reconstructed from multiple images captured manually around each plant. For reconstruction of 3D color point clouds from 2D color images, Agisoft Metashape Professional (Agisoft LLC, St. Petersburg, Russia) was employed. 

In the folder "PLANesT-3D_Images", 2D camera images used to construct the 3D point clouds of the PLANesT-3D dataset are provided.

The folder contains 2D camera images of 10 Pepper, 10 Rose and 14 Ribes plants.

The camera information files produced by Metashape for each reconstruction are also provided in the folders.

For the Ribes plants the following pairs of plants were extracted from the same scene reconstruction: 
- Ribes_02 and Ribes_12
- Ribes_04 and Ribes_13
- Ribes_05 and Ribes_14

They share the same images and the same camera information.
