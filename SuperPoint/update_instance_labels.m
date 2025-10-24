function instance_labels_2 = update_instance_labels(instance_labels_1,clusters_YY,solid_labels)

instance_labels_2 = instance_labels_1;

num_inst_labels = max(instance_labels_1);

IDX_solid = find(solid_labels);

labels_solid = clusters_YY(IDX_solid);

labels_ids = unique(labels_solid);

for c = 1:length(labels_ids)

    cc = labels_ids(c);

    num_inst_labels = num_inst_labels+1;

    instance_labels_2(clusters_YY==cc) = num_inst_labels;

end
