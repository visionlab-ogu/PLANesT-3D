function [superpoint_labels,points2D, labels2D] = extract_superpoints(ptCloud,gridStep,tsne_perp,min_dist_tsne)

ptCloudD = pcdownsample(ptCloud,'gridAverage',gridStep);
points2D = compute_TSNE(ptCloudD,tsne_perp);

[~,labels2D] = function_cluster_YY(points2D,min_dist_tsne);


KDTreeMdl = KDTreeSearcher(ptCloudD.Location);
[IdxMk,DMk] = knnsearch(KDTreeMdl,ptCloud.Location);

superpoint_labels = labels2D(IdxMk);