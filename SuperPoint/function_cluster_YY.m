function [branch_labels,final_labels] = function_cluster_YY(YY,min_dist_tsne)

show_figures = 0;


radius_tsne = 2*min_dist_tsne;

solid_thr = 0.8;
spec_thr = 0.0005;

min_pts = 50;


clusters_YY = TSNE_clusters(YY, min_dist_tsne);

if show_figures
    XYZ_YY = [YY zeros(size(YY,1),1)];
    ptCloud_YY = pointCloud(XYZ_YY);
    figure;
    colormap(hsv(max(double(clusters_YY+1))))
    pcshow(ptCloud_YY.Location,double(clusters_YY+1))
    pause
end

num_pts = size(YY,1);

branch_labels = cluster_local_props(YY,clusters_YY,radius_tsne);

if show_figures
    figure;
    colormap(hsv(max(double(branch_labels+1))))
    pcshow(ptCloud_YY.Location,double(branch_labels+1))
    title('Branches');
    pause
end

clusters_YY = TSNE_clusters_update(YY, clusters_YY, branch_labels, min_dist_tsne);

if show_figures
    figure;
    colormap(hsv(max(double(clusters_YY+1))))
    pcshow(ptCloud_YY.Location,double(clusters_YY+1))
    pause
end

instance_labels_1 = form_instance_labels_LF(YY, branch_labels, min_dist_tsne);



if max(clusters_YY(:)) > 1
cluster_props = cluster_boundaries(YY,clusters_YY);

Z = normalize_boundaries(YY,cluster_props);
solid_labels = check_solidity(Z,clusters_YY,num_pts,min_pts,solid_thr);
instance_labels_1 = update_instance_labels(instance_labels_1,clusters_YY,solid_labels);



clusters_YY = TSNE_clusters_update(YY, clusters_YY, instance_labels_1, min_dist_tsne);



num_not_labeled = sum(instance_labels_1==0);

while num_not_labeled > 0

    [newClusterY,solid_spect_1] = recluster_spectral(YY,clusters_YY,num_pts,min_pts,spec_thr);
    clusters_YY = newClusterY;

    instance_labels_1 = update_instance_labels(instance_labels_1,clusters_YY,solid_spect_1);
    clusters_YY = TSNE_clusters_update(YY, clusters_YY, instance_labels_1, min_dist_tsne);


    num_not_labeled = sum(instance_labels_1==0);

    if num_not_labeled == 0
        break
    end


    cluster_props = cluster_boundaries(YY,clusters_YY);
    Z = normalize_boundaries(YY,cluster_props);
    solid_labels = check_solidity(Z,clusters_YY,num_pts,min_pts,solid_thr);



    instance_labels_1 = update_instance_labels(instance_labels_1,clusters_YY,solid_labels);



    clusters_YY = TSNE_clusters_update(YY, clusters_YY, instance_labels_1, min_dist_tsne);


    num_not_labeled = sum(instance_labels_1==0);

    if show_figures
        figure;
        colormap(hsv(max(double(clusters_YY+1))))
        pcshow(ptCloud_YY.Location,double(clusters_YY+1))
        pause
    end


end

end

final_labels = instance_labels_1;


%     XYZ_YY = [YY zeros(size(YY,1),1)];
%     ptCloud = pointCloud(XYZ_YY);
%     fig_clus_YY = figure;
%     colormap(hsv(max(double(final_labels))))
%     pcshow(ptCloud.Location,double(final_labels))
% 
%     fig_clus_YY = figure;
%     colormap(hsv(max(double(branch_labels+2))))
%     pcshow(ptCloud.Location,double(branch_labels+2))