"""12/06/2023 Kaya Turgut"""

import numpy as np
import pickle
import open3d as o3d
import glob,re
import os
import h5py

dataset = "Pepper"
ply_dirs = "/media/visionlab/Data/Tübitak-1001-KT/data/cls/PLY_FILES_v2/"+dataset+"/*/"
ply_sub_dirs = sorted(glob.glob(ply_dirs),key=lambda x: (int(re.sub('\D','',x)),x))

for ply_sub_dir in ply_sub_dirs:
    print(ply_sub_dir)
    ply_files_leaf = sorted(glob.glob(os.path.join(ply_sub_dir, 'Leaf/*.ply')),key=lambda x: (int(re.sub('\D','',x)),x))
    ply_files_stem = sorted(glob.glob(os.path.join(ply_sub_dir, 'Stem/*.ply')),key=lambda x: (int(re.sub('\D','',x)),x))
    
    f_txt = open(os.path.join(ply_sub_dir, ply_sub_dir.split('/')[-2]+'.txt'), 'w')
    
    xyz, label = [], []
    for ply_file in ply_files_leaf:
        cloud = o3d.io.read_point_cloud(ply_file)
        points_array = np.array(cloud.points)
        colors_array =  np.array(cloud.colors)
        
        xyz.append(points_array)
        label.append(0)
        f_txt.write(ply_file+'\n')
        
    for ply_file in ply_files_stem:
        cloud = o3d.io.read_point_cloud(ply_file)
        points_array = np.array(cloud.points)
        colors_array =  np.array(cloud.colors)
        
        xyz.append(points_array)
        label.append(1)
        f_txt.write(ply_file+'\n')
    f_txt.close()       

    f_pickle = open(os.path.join(ply_sub_dir, ply_sub_dir.split('/')[-2]+'.pickle'), 'wb')
    pickle.dump(xyz, f_pickle, protocol=2)
    pickle.dump(label, f_pickle, protocol=2)
    f_pickle.close()
        
        
            


