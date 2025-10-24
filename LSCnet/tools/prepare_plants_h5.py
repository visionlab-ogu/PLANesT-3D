"""12/06/2023 Kaya Turgut"""

import numpy as np
import pickle
import open3d as o3d
import glob,re
import os
import h5py


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='uint8'):

    h5_fout = h5py.File(h5_filename,'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()
    
def pc_normalize_batch(pc_batch_data):
    for i in range(pc_batch_data.shape[0]):
        pc = pc_batch_data[i]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
        #m = np.max(np.sqrt(np.sum(pc**2, axis=-1)))
        pc = pc / m
        pc_batch_data[i] = pc
    return pc_batch_data

    
def sampling_random_one_cloud(cloud_array, sampling_num = 2048):
    pool_size = cloud_array.shape[0]
    if pool_size > sampling_num:
        choices = np.random.choice(pool_size, sampling_num, replace=False)
    else:
        choices = np.concatenate((np.random.choice(pool_size, pool_size, replace=False),
                                  np.random.choice(pool_size, sampling_num - pool_size, replace=True)))    
    return cloud_array[choices]



dataset = "Ribes"
ply_dirs = "/media/visionlab/Data/Tübitak-1001-KT/data/cls/PLY_FILES_v2/"+dataset
ply_sub_dirs = sorted(glob.glob(ply_dirs+"/*/"),key=lambda x: (int(re.sub('\D','',x)),x))

test_id = {"Pepper":[1,3,7], "Ribes":[3,10,11,14], "Rose":[1,3,9]}
test_xyz, test_label = [], []
train_xyz, train_label = [], []


f_merge_train_txt = open(os.path.join(ply_dirs, dataset +'_train.txt'), 'w')
f_merge_test_txt = open(os.path.join(ply_dirs, dataset +'_test.txt'), 'w')

for ply_sub_dir in ply_sub_dirs:
    print(ply_sub_dir)
    index = int(ply_sub_dir.split('_')[-1][:-1])
    
    f_txt = open(os.path.join(ply_sub_dir, ply_sub_dir.split('/')[-2]+'.txt'), 'r') 
    file = open(os.path.join(ply_sub_dir, ply_sub_dir.split('/')[-2]+'.pickle'), 'rb')
    
    xyz_list = pickle.load(file, encoding='latin1') # encoding keyword for python3
    labels_list = pickle.load(file, encoding='latin1') # encoding keyword for python3    

    if index in test_id[dataset]:
        #print(index)
        f_merge_test_txt.write(f_txt.read())
    else:
        f_merge_train_txt.write(f_txt.read())
    f_txt.close()
    
    for xyz, label in zip(xyz_list,labels_list):
        data_sampled = sampling_random_one_cloud(xyz,2048)  
      
        if index in test_id[dataset]:
            test_xyz.append(data_sampled)
            test_label.append(label)            
        else:
            train_xyz.append(data_sampled)
            train_label.append(label)    
 
f_merge_train_txt.close()
f_merge_test_txt.close() 

test_xyz_array = np.array(test_xyz)        
train_xyz_array = np.array(train_xyz)  
test_label_array = np.array(test_label)        
train_label_array = np.array(train_label) 

test_data = pc_normalize_batch(test_xyz_array) 
train_data = pc_normalize_batch(train_xyz_array) 


save_h5(os.path.join(ply_dirs, dataset +'_test.h5'),test_data,test_label_array)    
save_h5(os.path.join(ply_dirs, dataset +'_train.h5'),train_data,train_label_array)   





