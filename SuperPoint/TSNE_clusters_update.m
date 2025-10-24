function clusters_YY_new = TSNE_clusters_update(YY, clusters_YY, branch_labels, min_dist_tsne)

clusters_YY_new = zeros(size(YY,1),1);

IDX_branch = branch_labels == 0;
YY = YY(IDX_branch,:);

XYZ_YY = [YY zeros(size(YY,1),1)];
ptCloud = pointCloud(XYZ_YY);

clusters_YY_in = pcsegdist(ptCloud, min_dist_tsne);
clusters_YY_new(IDX_branch) = clusters_YY_in;

num_clusters_new = max(clusters_YY_new);

for c = 1:max(clusters_YY_new)
    IDX_c = find(clusters_YY_new == c);
    TT = clusters_YY(IDX_c);
    TT_U = unique(nonzeros(TT));
    
    if length(TT_U)>1
        for ci = 2:length(TT_U)
            num_clusters_new = num_clusters_new + 1;
            clusters_YY_new(IDX_c(TT==TT_U(ci))) = num_clusters_new;
        end
    end
end


% if not(isempty(clusters_YY_in))
% 
% fig_clus_YY = figure;
% numClusters = max(double(clusters_YY_in+2));
% colormap(hsv(numClusters))
% pcshow(ptCloud.Location,double(clusters_YY_in+2))
% title('Clusters of TSNE')
% 
% pause
% close all
% end