function Z = normalize_boundaries(YY,cluster_props)

num_clusters = length(cluster_props);
Z = struct;

% figY = figure;

for c = 1:num_clusters
    B_leaf_ind = cluster_props(c).boundary_indices;

    if not(isempty(B_leaf_ind))

    tip_index_final_1 = cluster_props(c).tip_index_1;
    tip_index_final_2 = cluster_props(c).tip_index_2;

    B_leaf = YY(B_leaf_ind,:);
    tip_1 = YY(tip_index_final_1,:);
    tip_2 = YY(tip_index_final_2,:);

    B_leaf_T = B_leaf - repmat(tip_1,size(B_leaf,1),1);
    tip_2 = tip_2 - tip_1;

    scale_factor = norm(tip_2);

    B_leaf_TS = B_leaf_T/scale_factor;
    u = tip_2/scale_factor;
    v = [u(2) -u(1)];

    B_leaf_TSR = B_leaf_TS*[u' v'];
    Z(c).normBoundary = B_leaf_TSR;

    else
        Z(c).normBoundary = [];
    end

%     figure(figY);
%     plot(B_leaf_TSR(:,1),B_leaf_TSR(:,2),'r.');
%     pause
end
   



