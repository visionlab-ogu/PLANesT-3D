'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
#import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
#sys.path.append(BASE_DIR)
#sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
#import modelnet_dataset
import modelnet_h5_dataset
import open3d as o3d

USE_FEAT = False
log_file = "log_scanObjectNN_background_RUM_II_cum_att_2nd_v2"
save_file = os.path.join(log_file,"dump_one")
sys.path.append(os.path.join(BASE_DIR, log_file))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default=os.path.join(log_file,"model.ckpt"), help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default=save_file, help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--data_idx', type=int, default=578)
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate_'+str(FLAGS.data_idx)+'.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 15
SHAPE_NAMES = [line.rstrip() for line in \
    open('/home/data/scanobjectNN_hdf5_2048/shape_names.txt')] 

HOSTNAME = socket.gethostname()

TEST_DATASET = modelnet_h5_dataset.DatasetOnePC('/home/data/scanobjectNN_hdf5_2048/test_files_eigen_bg.txt', npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(data_idx):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")
    
    """tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        if 'att' in var.name:
            print(var.name, val)  # Prints the name of the variable alongside its value."""


    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'end_points':end_points,
           'loss': total_loss}

    eval_one_epoch(sess, ops, data_idx)

def eval_one_epoch(sess, ops, data_idx=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    if USE_FEAT:
        cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    else:
        cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    batch_data, batch_label, batch_feat = TEST_DATASET.get_data(data_idx)
    bsize = batch_data.shape[0]
    if USE_FEAT:
        cur_batch_data[0:bsize,...] = np.concatenate((batch_data,batch_feat), axis=-1)
    else:
        cur_batch_data[0:bsize,...] = batch_data
        
    cur_batch_label[0:bsize] = batch_label

    feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                 ops['labels_pl']: cur_batch_label,
                 ops['is_training_pl']: is_training}
    loss_val, pred_val, end_points = sess.run([ops['loss'], ops['pred'], ops['end_points']], feed_dict=feed_dict)
    pred_val = np.argmax(pred_val, 1)
    
    log_string( "GT:" +str(SHAPE_NAMES[cur_batch_label[0]]))
    log_string( "Pred:" + str(SHAPE_NAMES[pred_val[0]]))

   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(end_points[2]['xyz'][0])
    o3d.io.write_point_cloud(os.path.join(save_file,"1st_fps_"+str(data_idx)+".pcd"), pcd)
    """pcd = o3d.geometry.PointCloud()    
    pcd.points = o3d.utility.Vector3dVector(end_points[2]['new_xyz'][0])
    o3d.io.write_point_cloud(os.path.join(save_file,"2nd_shift_xyz_"+str(data_idx)+".pcd"), pcd)"""
    pcd = o3d.geometry.PointCloud()   
    pcd.points = o3d.utility.Vector3dVector(end_points[2]['fps'][0])
    o3d.io.write_point_cloud(os.path.join(save_file,"2nd_fps_"+str(data_idx)+".pcd"), pcd)
   
    np.savetxt(os.path.join(save_file,"2nd_radius_update_"+str(data_idx)+".txt"), np.squeeze(end_points[2]['rad_update'],0))
    
 



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(data_idx=FLAGS.data_idx)
    LOG_FOUT.close()
