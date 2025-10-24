clear all
close all

DATA_FOLDER = '.\data\';

PlantName = 'Rose_03';

conf_thresh = 5;
gridStep = 0.12;
tsne_perp = 30;
min_dist_tsne = 1;


ptCloud = pcread([DATA_FOLDER PlantName '.ply']);
ptConfidence = load([DATA_FOLDER PlantName '_Confidence.txt']);

Idx_C = find(ptConfidence>conf_thresh);
ptCloud = select(ptCloud,Idx_C);


figure(1);
pcshow(ptCloud);
title('Point Cloud');


[superpoint_labels, points2D, labels2D]  = extract_superpoints(ptCloud,gridStep,tsne_perp,min_dist_tsne);

N = max(double(superpoint_labels));

XYZ_YY = [points2D zeros(size(points2D,1),1)];
ptCloud_2D = pointCloud(XYZ_YY);

figure(2);
colormap(hsv(N))
pcshow(ptCloud_2D.Location,labels2D)
title('2D Map with Superpoint Labels')


figure(3);
colormap(hsv(N))
pcshow(ptCloud.Location,superpoint_labels);
title('Point Cloud with Superpoint Labels')