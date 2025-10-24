function branch_labels = check_solidity(Z,clusters_YY,num_pts,min_pts,solid_thr)


num_clusters = length(Z);
branch_labels = zeros(num_pts,1);

for i = 1:num_clusters

    IDX_c = find(clusters_YY==i);

    if isempty(Z(i).normBoundary)
        branch_labels(IDX_c) = 1;
    else
        if length(IDX_c) < min_pts
            branch_labels(IDX_c) = 1;
        else
            B_cluster = Z(i).normBoundary;
            YY_I = 400*(B_cluster-min(B_cluster));
            M = ceil(max(YY_I(:,1)))+2;
            N = ceil(max(YY_I(:,2)))+2;
    
            YY_r_poly = [YY_I;YY_I(1,:)];
            bw = poly2mask(YY_r_poly(:,1)',YY_r_poly(:,2)',N,M);
            soli_c  = regionprops(bw, 'solidity');

            if length(soli_c) == 1
                if soli_c(1).Solidity > solid_thr
                    branch_labels(IDX_c) = 1;
                end
            end

%             fig_bw = figure;
%             imshow(bw);
%             soli_c
%             pause
%             close(fig_bw);

        end
    end
end

