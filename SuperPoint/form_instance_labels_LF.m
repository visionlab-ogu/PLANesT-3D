function instance_labels_1 = form_instance_labels_LF(YY, branch_labels, min_dist_tsne)

instance_labels_1 = zeros(size(YY,1),1);

IDX_branch = branch_labels == 1;
YY = YY(IDX_branch,:);

XYZ_YY = [YY zeros(size(YY,1),1)];
ptCloud = pointCloud(XYZ_YY);

clusters_YY_in = pcsegdist(ptCloud, min_dist_tsne);
instance_labels_1(IDX_branch) = clusters_YY_in;
