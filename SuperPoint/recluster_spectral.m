function [newClusterY,solid_spect_1] = recluster_spectral(YY,clusterY,num_pts,min_pts,spec_thr)


solid_spect_1 = zeros(num_pts,1);

num_clusters = max(clusterY);

newClusterY = clusterY;
new_num_clusters = num_clusters;

% figY = figure;

for c = 1:num_clusters
    
    

    ind_clusters_YY = find(clusterY==c);

    if length(ind_clusters_YY) < min_pts

        idx_spect = ones(length(ind_clusters_YY),1);

    else
        YY_c = YY(clusterY==c,:);

        [~,V_temp,D_temp] = spectralcluster(YY_c,20);
        % disp(D_temp);
        num_spec_clus = sum(abs(D_temp)<=spec_thr);
        % disp(num_spec_clus);

        if num_spec_clus == 1
            solid_spect_1(ind_clusters_YY) = 1;
        end

    
        idx_spect = spectralcluster(YY_c,num_spec_clus);

    end

%     scat_plot = {'.r','.g','.b','.m','.c','.y'};
%     fig_spect = figure;
%     for ss = 1:max(idx_spect)
%         hold on;
%         plot(YY_c(idx_spect==ss,1),YY_c(idx_spect==ss,2),scat_plot{mod(ss,6)+1});
%         axis equal
%     end

%     pause

%     close(fig_spect);

    num_clusters_cc = max(idx_spect);

    if num_clusters_cc > 1
        for cc = 2:num_clusters_cc
            new_num_clusters = new_num_clusters+1;
            newClusterY(ind_clusters_YY(idx_spect==cc)) = new_num_clusters;
        end
    end




end
   



