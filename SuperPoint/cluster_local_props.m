function branch_labels = cluster_local_props(YY,clusterY,radius)



clusterY = double(clusterY);
num_clusters = max(clusterY);
num_points = size(YY,1);

branch_labels = zeros(num_points,1);

% fig_X = figure;
% plot(YY(:,1),YY(:,2),'g.');
% axis equal;
% pause
% close all

for c = 1:num_clusters

    IDX_c = find(clusterY==c);


    if and((length(IDX_c)>3),(length(IDX_c)<60000))

    Y_cluster = YY(IDX_c,:);
    B_leaf_ind_loc = boundary(Y_cluster(:,1),Y_cluster(:,2),0.7);
    
    ratio_ra = size(Y_cluster,1)/size(B_leaf_ind_loc,1);



    if ratio_ra < 3
        branch_labels(IDX_c) = 1;
    else
        

        num_points_c = length(IDX_c);

        DIST_pts = pdist(Y_cluster);
        DIST_pts = squareform(DIST_pts);

        for p = 1:num_points_c

            %point1 = Y_cluster(p,:);
            IDX_close = find(DIST_pts(p,:)<radius);
            points_neig = Y_cluster(IDX_close,:);
            covMat = cov(points_neig);

            [~,lambda_s] = eig(covMat,'vector');

            Lin_point = lambda_s(1)/lambda_s(2);

            if Lin_point < 0.1

                branch_labels(IDX_c(p)) = 1;
            else
                branch_labels(IDX_c(p)) = 0;
            end

        end
    end

    end
end





%     B_leaf = YY(B_leaf_ind,:);
%     B_dist = pdist(B_leaf);
%     B_dist = squareform(B_dist);
%     [MMM,ind_tip_1] = max(B_dist);
%     [~,ind_tip_2] = max(MMM);
%     ind_tip_2 = ind_tip_2(1);
%     ind_tip_1 = ind_tip_1(ind_tip_2);
% 
%     tip_index_final_1 = B_leaf_ind(ind_tip_1);
%     tip_index_final_2 = B_leaf_ind(ind_tip_2);
% 
%     cluster_props(c).tip_index_1 = tip_index_final_1;
%     cluster_props(c).tip_index_2 = tip_index_final_2;


%     % Plotting
% 
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
% 



end




