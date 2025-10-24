function cluster_props = cluster_boundaries(YY,clusterY)

clusterY = double(clusterY);
num_clusters = max(clusterY);

cluster_props = struct;

for c = 1:num_clusters

    IDX_c = find(clusterY==c);

    Y_cluster = YY(IDX_c,:);
    B_leaf_ind_loc = boundary(Y_cluster(:,1),Y_cluster(:,2),0.8);
    B_leaf_ind = IDX_c(B_leaf_ind_loc);

    cluster_props(c).boundary_indices = B_leaf_ind;

    if not(isempty(B_leaf_ind))

    B_leaf = YY(B_leaf_ind,:);
    B_dist = pdist(B_leaf);
    B_dist = squareform(B_dist);
    [MMM,ind_tip_1] = max(B_dist);
    [~,ind_tip_2] = max(MMM);
    ind_tip_2 = ind_tip_2(1);
    ind_tip_1 = ind_tip_1(ind_tip_2);

    tip_index_final_1 = B_leaf_ind(ind_tip_1);
    tip_index_final_2 = B_leaf_ind(ind_tip_2);

    cluster_props(c).tip_index_1 = tip_index_final_1;
    cluster_props(c).tip_index_2 = tip_index_final_2;

    else
        cluster_props(c).tip_index_1 = [];
        cluster_props(c).tip_index_2 = [];
    end


end

% fig_X = figure;
% plot(YY(:,1),YY(:,2),'g.');
% axis equal;
% 
% for c = 1:num_clusters
%     B_leaf_ind = cluster_props(c).boundary_indices;
%     tip_index_final_1 = cluster_props(c).tip_index_1;
%     tip_index_final_2 = cluster_props(c).tip_index_2;
% 
%     B_leaf = YY(B_leaf_ind,:);
%     tip_1 = YY(tip_index_final_1,:);
%     tip_2 = YY(tip_index_final_2,:);
% 
%     figure(fig_X);
%     hold on;
%     plot(B_leaf(:,1),B_leaf(:,2),'r.');
%     plot(tip_1(1),tip_1(2),'*g');
%     plot(tip_2(1),tip_2(2),'*g');
%     axis equal
% 
%     pause
% end

