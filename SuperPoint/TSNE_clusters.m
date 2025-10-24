function clusters_YY = TSNE_clusters(YY, min_dist_tsne)

XYZ_YY = [YY zeros(size(YY,1),1)];
ptCloud = pointCloud(XYZ_YY);

clusters_YY = pcsegdist(ptCloud, min_dist_tsne);

% fig_clus_YY = figure;
% numClusters = max(double(clusters_YY));
% colormap(hsv(numClusters))
% pcshow(ptCloud.Location,clusters_YY)
% title('Clusters of TSNE')