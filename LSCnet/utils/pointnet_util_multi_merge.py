""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point, query_ball_point_random, query_ball_point_pointwise
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import math

def _gauss(xyz_rel, sigma=1.0):
    return tf.exp(-0.5*tf.square(xyz_rel)/sigma**2)/(sigma*((2*math.pi)**0.5))

def sample_and_group_for_modules(npoint, radius, nsample, xyz, new_xyz, points): 
    B, N, C = xyz.get_shape().as_list() 
    rand_id = np.tile(np.arange(N).astype('int32'),(B,npoint,1))
    shuffle_idx = np.random.rand(*rand_id.shape).argsort(axis=-1)
    rand_id_shuffle = tf.constant(np.take_along_axis(rand_id,shuffle_idx, axis=-1))
    
    idx_rad, cnt = query_ball_point_random(radius, nsample, xyz, new_xyz, rand_id_shuffle) 
    """_,idx_knn = knn_point(nsample, xyz, new_xyz)     
    cond = tf.equal(cnt, tf.constant(0, dtype=tf.int32))
    idx_rad_new = tf.where(tf.tile(tf.expand_dims(cond,-1), [1,1,nsample]), idx_knn, idx_rad)""" 
    
    grouped_xyz = group_point(xyz, idx_rad) # (batch_size, npoint, nsample, 3)
    grouped_points = group_point(points, idx_rad) # (batch_size, npoint, nsample, channel)
    return grouped_xyz, grouped_points

def combined_module(npoint, radius, nsample, xyz, points, use_xyz, bn_decay, is_training, scope, center_flag, radius_flag):     
    if scope == 'layer1' and points is None:
        idx_fl, _ = query_ball_point(radius, nsample, xyz, xyz)
        grouped_xyz_fl = group_point(xyz, idx_fl) # (batch_size, npoint, nsample, 3)
        grouped_xyz_fl -= tf.tile(tf.expand_dims(xyz, 2), [1,1,nsample,1]) # translation normalization
        feat_fl = tf_util.conv2d(grouped_xyz_fl, 32, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_adap_fl_feat', bn_decay=bn_decay,
                            data_format='NHWC') # (B,N,nsample,32)
        feat_fl = tf_util.conv2d(feat_fl, 32, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_adap_fl_feat_2', bn_decay=bn_decay,
                            data_format='NHWC') # (B,N,nsample,32)
        points =  tf.reduce_max(feat_fl, axis=[2], name='maxpool') # (B,N,32)  
    """    points_fl =  tf.reduce_max(feat_fl, axis=[2], name='maxpool') # (B,N,32)  
    else:
        points_fl = points"""
    
    fps_idx = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, fps_idx) # (batch_size, npoint, 3)
    new_points = gather_point(points, fps_idx) # (batch_size, npoint, C)
    sa_feat = None
        
    if center_flag==True and radius_flag==False:
        shifted_xyz, shift_val, sa_feat = center_shift_module(npoint, radius, nsample, xyz, new_xyz, points, new_points, bn_decay, is_training, scope, method='CSA-I')
        radius_adap = tf.constant(radius)
        idx_adap, cnt = query_ball_point(radius, nsample, xyz, shifted_xyz)
        
        """_,idx_adap_knn = knn_point(nsample, xyz, shifted_xyz)     
        cond = tf.equal(cnt, tf.constant(0, dtype=tf.int32))
        idx_adap_new = tf.where(tf.tile(tf.expand_dims(cond,-1), [1,1,nsample]), idx_adap_knn, idx_adap)"""  
            
        grouped_xyz_adap = group_point(xyz, idx_adap) # (batch_size, npoint, nsample, 3)
        grouped_xyz_adap -= tf.tile(tf.expand_dims(shifted_xyz, 2), [1,1,nsample,1]) # translation normalization        
        if sa_feat is None:
            grouped_points_adap = group_point(points, idx_adap) # (batch_size, npoint, nsample, channel) #Activate when CSI-II and CSI-I
        else:
            grouped_points_adap = tf.concat([tf.tile(sa_feat, [1,1,nsample,1]), group_point(points, idx_adap)], axis=-1) # (batch_size, npoint, nsample, 2*channel)
               
    elif center_flag==False and radius_flag==True:
        radius_adap = radius_update_module(npoint, radius, nsample, xyz, new_xyz, points, new_points, bn_decay, is_training, scope, agg_type='max_pool', module='None')
        shifted_xyz = new_xyz
        shift_val = tf.constant(0.0)
        radius_adap_lim = tf.maximum(1e-6, radius_adap)
        idx_adap, cnt = query_ball_point_pointwise(radius_adap_lim, nsample, xyz, new_xyz)
        
        """_,idx_adap_knn = knn_point(nsample, xyz, new_xyz)     
        cond = tf.equal(cnt, tf.constant(0, dtype=tf.int32))
        idx_adap_new = tf.where(tf.tile(tf.expand_dims(cond,-1), [1,1,nsample]), idx_adap_knn, idx_adap)"""   
                
        grouped_xyz_adap = group_point(xyz, idx_adap) # (batch_size, npoint, nsample, 3)
        grouped_xyz_adap -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
        grouped_points_adap = group_point(points, idx_adap) # (batch_size, npoint, nsample, channel)
   
    else:
        shifted_xyz, shift_val, sa_feat = center_shift_module(npoint, radius, nsample, xyz, new_xyz, points, new_points, bn_decay, is_training, scope, method='CSA-I') 
        #_,idx_knn = knn_point(1, xyz, shifted_xyz)
        #shifted_xyz = tf.squeeze(group_point(xyz, idx_knn),2)
        radius_adap = radius_update_module(npoint, radius, nsample, xyz, shifted_xyz, points, tf.squeeze(sa_feat,2), bn_decay, is_training, scope, agg_type='max_pool', module='None')
        #radius_adap = radius_update_module(npoint, radius, nsample, xyz, shifted_xyz, points, new_points, bn_decay, is_training, scope, agg_type='max_pool', module='None')
        radius_adap_lim = tf.maximum(1e-6, radius_adap)
        idx_adap, cnt = query_ball_point_pointwise(radius_adap, nsample, xyz, shifted_xyz) 
        
        """_,idx_adap_knn = knn_point(nsample, xyz, shifted_xyz)     
        cond = tf.equal(cnt, tf.constant(0, dtype=tf.int32))
        idx_adap_new = tf.where(tf.tile(tf.expand_dims(cond,-1), [1,1,nsample]), idx_adap_knn, idx_adap)"""  
                                  
        grouped_xyz_adap = group_point(xyz, idx_adap) # (batch_size, npoint, nsample, 3)
        grouped_xyz_adap -= tf.tile(tf.expand_dims(shifted_xyz, 2), [1,1,nsample,1]) # translation normalization
        grouped_points_adap = tf.concat([tf.tile(sa_feat, [1,1,nsample,1]), group_point(points, idx_adap)], axis=-1) # (batch_size, npoint, nsample, channel)
        
        if sa_feat is None:
            grouped_points_adap = group_point(points, idx_adap) # (batch_size, npoint, nsample, channel) #Activate when CSI-II and CSI-I
        else:
            grouped_points_adap = tf.concat([tf.tile(sa_feat, [1,1,nsample,1]), group_point(points, idx_adap)], axis=-1) # (batch_size, npoint, nsample, 2*channel)
        
    if use_xyz:
        new_points_adap = tf.concat([grouped_xyz_adap, grouped_points_adap], axis=-1) # (batch_size, npoint, nample, 3+channel)
    else:
        new_points_adap = grouped_points_adap
            
    return shifted_xyz, new_points_adap, idx_adap, grouped_xyz_adap, new_xyz, sa_feat, shift_val, radius_adap
        

def center_shift_module(npoint, radius, nsample, xyz, center_xyz, points, center_points, bn_decay, is_training, scope, method):       
    grouped_xyz, grouped_points = sample_and_group_for_modules(npoint, 2*radius, nsample, xyz, center_xyz, points)  
    grouped_xyz_rel = grouped_xyz - tf.tile(tf.expand_dims(center_xyz, 2), [1,1,nsample,1]) # translation normalization
    grouped_points_rel = grouped_points - tf.tile(tf.expand_dims(center_points, 2), [1,1,nsample,1]) # translation normalization
    
    if method == 'CSA-I':
        shift, sa_feat = method_CSA_I(center_points, grouped_points, grouped_xyz_rel, radius, bn_decay, is_training)     
    elif method == 'CSA-II':
        shift, sa_feat = method_CSA_II(center_points, grouped_points, grouped_xyz_rel, radius, bn_decay, is_training, relation = 'concat')
    elif method == 'CSA-III':
        shift, sa_feat = method_CSA_III(center_points, grouped_points, center_xyz, grouped_xyz_rel, radius, bn_decay, is_training, knn=True) 
    elif method == 'CSI-I':
        shift, sa_feat = method_CSI_I(center_points, grouped_points, center_xyz, radius, bn_decay, is_training)      
    elif method == 'CSI-II':
        shift, sa_feat = method_CSI_II(grouped_points, grouped_xyz, radius, bn_decay, is_training)

        
    shifted_xyz = tf.add(center_xyz,shift)    
    #_,idx_knn = knn_point(1, xyz, shifted_xyz)        
    #shifted_xyz_nn = tf.squeeze(group_point(xyz, idx_knn), axis=-2)    
    return shifted_xyz, shift, sa_feat
        
  
def radius_update_module(npoint, radius, nsample, xyz, center_xyz, points, center_points, bn_decay, is_training, scope, agg_type, module):      
    shell = False
    T = 4
    rad_local = 2*radius
                  
    grouped_xyz, grouped_points = sample_and_group_for_modules(npoint, rad_local, nsample, xyz, center_xyz, points)  
    grouped_xyz_rel = grouped_xyz - tf.tile(tf.expand_dims(center_xyz, 2), [1,1,nsample,1]) # translation normalization
    grouped_points_rel = grouped_points - tf.tile(tf.expand_dims(center_points, 2), [1,1,nsample,1]) # translation normalization  
    B, M, nsample, C = grouped_points_rel.get_shape().as_list()
    feat_out_qk = C//1
    feat_out_v = C//1
        
    rel_feat = tf_util.conv2d(grouped_points_rel, C, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='conv_feat_rel', bn_decay=bn_decay,
                                data_format='NHWC') # (B,M,nsample,Ch)        
    rel_dist = tf.sqrt(tf.reduce_sum(tf.square(grouped_xyz_rel), axis=-1)) # B,M,nsample         
            
    Rt_list = [] # B,M,T,Ch
    start = 0.0 
    
    
    for t in range(1,T+1):
        end = start + rad_local * (float(t)/T)
        pindex_inradius = tf.cast(tf.logical_and(rel_dist >= start, rel_dist < end), dtype=tf.float32) #B,M,nsample
        if agg_type == "weighted_cum":  
            print agg_type
            Rt_list.append(tf.reduce_sum(tf.multiply(rel_feat,tf.expand_dims(pindex_inradius,-1)), axis=-2)/(1e-6 + tf.cast(tf.reduce_sum(pindex_inradius, keep_dims=True, axis=-1), dtype=tf.float32)))# B,M,Ch
        elif agg_type == "max_pool":            
            print agg_type
            Rt_list.append(tf.reduce_max(tf.multiply(rel_feat,tf.expand_dims(pindex_inradius, -1)), axis=-2)) # B,M,Ch         
                
        if shell:    
            start = end #Multi-scale or shell             
            
    Rt = tf.stack(Rt_list, 2)        
    if module == "attention":  
        print "attention module"
        feat_q = tf_util.conv2d(Rt, feat_out_qk, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_q_rad', use_bias=False,
                                 bn_decay=bn_decay, activation_fn=None) #(B,M,1,feat_out_qk)
        feat_k = tf_util.conv2d(Rt, feat_out_qk, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_k_rad', use_bias=False,
                                 bn_decay=bn_decay, activation_fn=None)
        feat_v = tf_util.conv2d(Rt, feat_out_v, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_v_rad', use_bias=True,
                                 bn_decay=bn_decay, activation_fn=None) #(B,M,nsample,feat_out_v)
        
        weight = tf.nn.softmax(tf.matmul(feat_q, tf.transpose(feat_k,(0,1,3,2)))/tf.sqrt(tf.cast(feat_out_qk, tf.float32)), dim=-1) #(B,M,T,T)
        sa_feat = tf.matmul(weight,feat_v)  # B,M,T,T * B,M,T,Ch -> (B,M,T,Ch)  
        Rt = Rt + sa_feat # (B,M,T,Ch)   
    delta_rad = tf_util.conv2d(Rt, 1, [1,T],
                            padding='VALID', stride=[1,1],
                            bn=False, is_training=is_training,
                            scope='conv_delta_rad', bn_decay=bn_decay,
                            data_format='NHWC',activation_fn=tf.nn.tanh) # (B,M,1,1)
    radius_adap = tf.add(radius, tf.squeeze(delta_rad,2)) #B,M,1   
    return radius_adap
    
def method_CSA_I(center_points, group_feat_local, group_xyz_rel_local, radius, bn_decay, is_training):
    print '********************(method_CSA_I)********************'
    B, M, nsample, C = group_feat_local.get_shape().as_list() #B,N,M,C+3 if use_xyz == True    
    feat_out_qk = C//2
    feat_out_v = C//2
    
    feat_q = tf_util.conv2d(tf.expand_dims(center_points, axis=2), feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_q', use_bias=False, 
                             activation_fn=None) #(B,M,1,feat_out_qk)
    feat_k = tf_util.conv2d(group_feat_local, feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_k', use_bias=False,
                             activation_fn=None)
    feat_v = tf_util.conv2d(group_feat_local, feat_out_v, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_v', use_bias=True,
                             activation_fn=None) #(B,M,nsample,feat_out_v)
    
    weight = tf.matmul(tf.tile(feat_q, [1,1,nsample,1]), feat_k, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,nsample,nsample)
    weight = tf.nn.softmax(weight, dim=-1) 
    sa_feat = tf.reduce_sum(tf.matmul(weight,feat_v),axis=2) # (B,M,C_v) = B,M,nsample,nsample * B,M,nsample,C_v -> (B,M,C_v) 
    
    sa_feat = tf_util.conv1d(sa_feat, C, 1,
                             padding='VALID', stride=1,
                             bn=True, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_encod', use_bias=True,
                             activation_fn=tf.nn.relu) # (B,M,1,C) 
    sa_feat = tf.expand_dims(sa_feat + center_points, axis=2) # (B,M,1, C)
    
    group_feat_edge = group_feat_local - tf.tile(sa_feat, [1,1,nsample,1]) # (B,M,nsample,C)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 32, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_trans_1', use_bias=True,
                             activation_fn=tf.nn.relu)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_trans_2', use_bias=False,
                             activation_fn=tf.nn.tanh) # (B,M,nsample,3)
    
    delta_xyz = tf.reduce_mean(tf.multiply(group_feat_edge,group_xyz_rel_local), axis=2) # (B,M,3)
    return delta_xyz, sa_feat


def method_CSA_III(center_points, group_feat_local, center_xyz, group_xyz_rel_local, radius, bn_decay, is_training, knn):
    print '********************(method_CSA_III)********************'
    B, M, nsample, C = group_feat_local.get_shape().as_list() #B,N,M,C+3 if use_xyz == True    
    feat_out_qk = C//2
    feat_out_v = C//2
    
    
    feat_q = tf_util.conv2d(tf.expand_dims(center_points, axis=2), feat_out_qk, [1,1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_q', use_bias=False,
                             activation_fn=None) #(B,M,1,feat_out_qk)
    feat_k = tf_util.conv2d(group_feat_local, feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_k', use_bias=False,
                             activation_fn=None)
    feat_v = tf_util.conv2d(group_feat_local, feat_out_v, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_v', use_bias=True,
                             activation_fn=None) #(B,M,nsample,feat_out_v)
    
    weight = tf.matmul(tf.tile(feat_q, [1,1,nsample,1]), feat_k, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,nsample,nsample)
    weight = tf.nn.softmax(weight, dim=-1) 
    sa_feat = tf.reduce_sum(tf.matmul(weight,feat_v),axis=2) # (B,M,C_v) = B,M,nsample,nsample * B,M,nsample,C_v
    
    sa_feat = tf_util.conv1d(sa_feat, C, 1,
                         padding='VALID', stride=1,
                         bn=True, is_training=is_training, bn_decay=bn_decay,
                         scope='conv_encod', use_bias=True,
                         activation_fn=tf.nn.relu) # (B,M,C) 
    sa_feat = sa_feat + center_points # (B,M,C)
    
    if knn==True:
        K = 8
        _,idx_nl = knn_point(K+1, center_xyz, center_xyz)
        grouped_points_nl = group_point(sa_feat, idx_nl[:,:,1:]) # (batch_size, npoint, K, channel)
    
        feat_q_nl = tf_util.conv2d(tf.expand_dims(sa_feat, axis=2), feat_out_qk, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_q_nl', use_bias=False,
                                 activation_fn=None) #(B,M,1,feat_out_qk)
        feat_k_nl  = tf_util.conv2d(grouped_points_nl, feat_out_qk, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_k_nl', use_bias=False,
                                 activation_fn=None)  #(B,M,K,feat_out_qk)
        feat_v_nl  = tf_util.conv2d(grouped_points_nl, feat_out_v, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=False, is_training=is_training,
                                 scope='conv_att_v_nl', use_bias=True,
                                 activation_fn=None) #(B,M,K,feat_out_qk)
        
        weight_nl = tf.matmul(tf.tile(feat_q_nl, [1,1,K,1]), feat_k_nl, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,K,K)
        weight_nl = tf.nn.softmax(weight_nl, dim=-1) 
        sa_feat_nl = tf.reduce_sum(tf.matmul(weight_nl,feat_v_nl),axis=2) # (B,M,C)

    else:
        feat_q_nl = tf_util.conv1d(sa_feat, feat_out_qk, 1,
                                 padding='VALID', stride=1,
                                 bn=False, is_training=is_training,
                                 scope='conv_att_q_nl', use_bias=False,
                                 activation_fn=None) #(B,M,feat_out_qk)
        feat_k_nl  = tf_util.conv1d(sa_feat, feat_out_qk, 1,
                                 padding='VALID', stride=1,
                                 bn=False, is_training=is_training,
                                 scope='conv_att_k_nl', use_bias=False,
                                 activation_fn=None)
        feat_v_nl  = tf_util.conv1d(sa_feat, feat_out_v, 1,
                                 padding='VALID', stride=1,
                                 bn=False, is_training=is_training,
                                 scope='conv_att_v_nl', use_bias=True,
                                 activation_fn=None) #(B,M,feat_out_v)
    
        weight_nl = tf.matmul(feat_q_nl, feat_k_nl, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,M)
        weight_nl = tf.nn.softmax(weight_nl, dim=-1) 
        sa_feat_nl = tf.matmul(weight_nl,feat_v_nl) # (B,M,C)
    
    sa_feat_nl = tf_util.conv1d(sa_feat_nl, C, 1,
                         padding='VALID', stride=1,
                         bn=True, is_training=is_training, bn_decay=bn_decay,
                         scope='conv_encod_nl', use_bias=True,
                         activation_fn=tf.nn.relu) # (B,M,C) 
    sa_feat_nl = sa_feat_nl + sa_feat # (B,M,C)
    

    group_feat_edge = group_feat_local - tf.tile(tf.expand_dims(sa_feat_nl, axis=2), [1,1,nsample,1]) # (B,M,nsample,C)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 32, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_trans_1', use_bias=True,
                             activation_fn=tf.nn.relu)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_trans_2', use_bias=False,
                             activation_fn=tf.nn.tanh) # (B,M,nsample,3)
    
    delta_xyz = tf.reduce_mean(tf.multiply(group_feat_edge,group_xyz_rel_local), axis=2) # (B,M,3)
    return delta_xyz, tf.expand_dims(sa_feat_nl, axis=2)

def method_CSA_II(center_points, group_feat_local, group_xyz_rel_local, radius, bn_decay, is_training, relation):
    print '********************(method_CSA_II)********************'
    B, M, nsample, C = group_feat_local.get_shape().as_list() 
    r = 2
    feat_out_qk = C//r
    feat_out_v = C//r
    
    feat_q = tf_util.conv2d(tf.expand_dims(center_points, axis=2), feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_q', use_bias=False,
                             activation_fn=None) #(B,M,1,feat_out_qk)
    feat_k = tf_util.conv2d(group_feat_local, feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_k', use_bias=False,
                             activation_fn=None)
    feat_v = tf_util.conv2d(group_feat_local, feat_out_v, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_v', use_bias=True,
                             activation_fn=None) #(B,M,nsample,feat_out_v)
    pos_lin = tf_util.conv2d(group_xyz_rel_local, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_pos_lin', use_bias=True,
                             activation_fn=None) #(B,M,nsample,3)
    if relation == 'sum':
        print '********************'
        print relation
        rel_func = tf.tile(feat_q, [1,1,nsample,1]) + feat_k              
        
    elif relation == 'sub':
        print '********************'
        print relation
        rel_func = tf.tile(feat_q, [1,1,nsample,1]) - feat_k
    
    elif relation == 'hadamard':
        print '********************'
        print relation
        rel_func = tf.tile(feat_q, [1,1,nsample,1]) * feat_k
        
    elif relation == 'dot':
        print '********************'
        print relation
        rel_func = tf.squeeze(tf.matmul(tf.expand_dims(tf.tile(feat_q, [1,1,nsample,1]) ,-2),tf.expand_dims(feat_k,-1)),-1) # (B,M,nsample,1)    
        
    elif relation == 'concat':      
        print '********************'
        print relation
        rel_func = tf.concat([tf.tile(feat_q, [1,1,nsample,1]) ,feat_k],-1)       
        
    att_mlp = tf_util.conv2d(tf.concat([pos_lin,rel_func],-1), feat_out_v*r, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_att_mlp_1', use_bias=True,
                             activation_fn=tf.nn.relu) #(B,M,nsample,channel*att_dim_mlp) 
    att_mlp = tf_util.conv2d(att_mlp, feat_out_v, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_mlp_2', use_bias=True,
                             activation_fn=None) #(B,M,nsample,channel*att_dim_mlp)         
    sa_feat = tf.reduce_sum(tf.nn.softmax(att_mlp, dim=2)*feat_v, axis=2) # (B,M,C//r) Softmaz is added according to the Point Transformer
        
    sa_feat = tf.nn.relu(tf_util.batch_norm_for_conv2d(tf.expand_dims(sa_feat,2), is_training=is_training,
                                                       scope='bn', bn_decay=bn_decay, data_format='NHWC')) # (B,M,1, C//r) 
    sa_feat = tf_util.conv2d(sa_feat, C, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv_feat_fim', use_bias=True,
                                 activation_fn=None) #(B,M,1,C)
    
    sa_feat = sa_feat + tf.expand_dims(center_points,axis=2) 
       
    group_feat_edge = group_feat_local - tf.tile(sa_feat, [1,1,nsample,1]) # (B,M,nsample,C)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 32, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,  bn_decay=bn_decay,
                             scope='conv_trans_1', use_bias=True,
                             activation_fn=tf.nn.relu)
    group_feat_edge = tf_util.conv2d(group_feat_edge, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_trans_2', use_bias=False,
                             activation_fn=tf.nn.tanh) # (B,M,nsample,3)
    
    delta_xyz = tf.reduce_mean(tf.multiply(group_feat_edge,group_xyz_rel_local), axis=2) # (B,M,3)
    return delta_xyz, sa_feat



def method_CSI_I(center_points, group_feat_local, center_xyz, radius, bn_decay, is_training):
    print '********************(method_CSI_I)********************'
    B, M, nsample, C = group_feat_local.get_shape().as_list() #B,N,M,C+3 if use_xyz == True    
    feat_out_qk = C//2
    feat_out_v = C//2
    
    feat_q = tf_util.conv2d(tf.expand_dims(center_points, axis=2), feat_out_qk, [1,1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_q', use_bias=False,
                             activation_fn=None) #(B,M,1,feat_out_qk)
    feat_k = tf_util.conv2d(group_feat_local, feat_out_qk, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_k', use_bias=False,
                             activation_fn=None)
    feat_v = tf_util.conv2d(group_feat_local, feat_out_v, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_att_v', use_bias=True,
                             activation_fn=None) #(B,M,nsample,feat_out_v)
    
    weight = tf.matmul(tf.tile(feat_q, [1,1,nsample,1]), feat_k, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,nsample,nsample)
    weight = tf.nn.softmax(weight, dim=-1) 
    sa_feat = tf.reduce_sum(tf.matmul(weight,feat_v),axis=2) # (B,M,C_v) = B,M,nsample,nsample * B,M,nsample,C_v
    
    sa_feat = tf_util.conv1d(sa_feat, C, 1,
                         padding='VALID', stride=1,
                         bn=True, is_training=is_training, bn_decay=bn_decay,
                         scope='conv_encod', use_bias=True,
                         activation_fn=tf.nn.relu) # (B,M,C) 
    sa_feat = sa_feat + center_points # (B,M,C)
       
    
    feat_q_nl = tf_util.conv1d(sa_feat, feat_out_qk, 1,
                             padding='VALID', stride=1,
                             bn=False, is_training=is_training,
                             scope='conv_att_q_nl', use_bias=False,
                             activation_fn=None) #(B,M,feat_out_qk)
    feat_k_nl  = tf_util.conv1d(sa_feat, feat_out_qk, 1,
                             padding='VALID', stride=1,
                             bn=False, is_training=is_training,
                             scope='conv_att_k_nl', use_bias=False,
                             activation_fn=None)
    feat_v_nl  = tf_util.conv1d(sa_feat, feat_out_v, 1,
                             padding='VALID', stride=1,
                             bn=False, is_training=is_training,
                             scope='conv_att_v_nl', use_bias=True,
                             activation_fn=None) #(B,M,feat_out_v)
    
    weight_nl = tf.matmul(feat_q_nl, feat_k_nl, transpose_b=True)/tf.sqrt(tf.cast(feat_out_qk, tf.float32)) #(B,M,M)
    weight_nl = tf.nn.softmax(weight_nl, dim=-1) 
    sa_feat_nl = tf.matmul(weight_nl,feat_v_nl) # (B,M,C)

    sa_feat_nl = tf_util.conv1d(sa_feat_nl, C, 1,
                         padding='VALID', stride=1,
                         bn=True, is_training=is_training, bn_decay=bn_decay,
                         scope='conv_encod_nl', use_bias=True,
                         activation_fn=tf.nn.relu) # (B,M,C) 
    sa_feat_nl = sa_feat_nl + sa_feat # (B,M,C)
    
    xyz_sim = tf.expand_dims(center_xyz,axis=2) - tf.expand_dims(center_xyz,axis=1) # (B,M,M,3)
    feat_sim = tf.expand_dims(sa_feat_nl,axis=2) - tf.expand_dims(sa_feat_nl,axis=1) # (B,M,M,C)

    feat_sim = tf_util.conv2d(feat_sim, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_trans_1', use_bias=True,
                             activation_fn=tf.nn.relu)
    feat_sim = tf_util.conv2d(feat_sim, 3*3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_trans_2', use_bias=False,
                             activation_fn=None) # (B,M,M,9)

    shift = tf.einsum('iklxj,ikljy->iklxy', tf.expand_dims(xyz_sim,axis=-2), tf.reshape(feat_sim, [B,M,M,3,3])) #B,M,M,1,3
    shift = tf.reduce_mean(tf.squeeze(shift,-2), axis=2) #B,M,3  
    return shift, tf.expand_dims(sa_feat_nl, axis=2)   


def method_CSI_II(group_feat_local, group_xyz_local, radius, bn_decay, is_training):
    print '********************(method_CSI_II)********************'
    B, M, nsample, C = group_feat_local.get_shape().as_list()         
    
    diff_feat = tf.expand_dims(tf.transpose(group_feat_local,[0,1,3,2]), axis=-1) - tf.expand_dims(tf.transpose(group_feat_local,[0,1,3,2]), axis=-2) # (B,M,C,nsample,nsample)
    diff_feat = tf.linalg.set_diag(diff_feat, tf.transpose(group_feat_local,[0,1,3,2]))   #Add original feat to diagonal
   
    diff_xyz = tf.expand_dims(tf.transpose(group_xyz_local,[0,1,3,2]), axis=-1) - tf.expand_dims(tf.transpose(group_xyz_local,[0,1,3,2]), axis=-2) # (B,M,3,nsample,nsample)
    diff_xyz = tf.linalg.set_diag(diff_xyz, tf.transpose(group_xyz_local,[0,1,3,2])) #Add original coordinate to diagonal
   
    diff_feat = tf_util.conv2d(tf.reshape(diff_feat, [B*M,C,nsample,nsample]), 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training, bn_decay=bn_decay,
                             scope='conv_adap_center_1', use_bias=True,
                             data_format='NCHW', activation_fn=tf.nn.relu)
    diff_feat = tf_util.conv2d(diff_feat, 3*3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='conv_adap_center_2', use_bias=False,
                             data_format='NCHW', activation_fn=None) # (BM,9,nsample,nsample)

    shift = tf.einsum('ixjkl,ijykl->ixykl', tf.reshape(diff_xyz, [B*M,1,3,nsample,nsample]), tf.reshape(diff_feat, [B*M,3,3,nsample,nsample])) #(BM,1,3,nsample,nsample)
    shift = tf.reduce_mean(tf.squeeze(shift,1), axis=-1) #(BM,3,nsample)
    shift = tf.reshape(tf.reduce_max(shift, axis=-1), [B,M,3])
    return shift, None


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, use_center_shift=False, use_radius_update=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        shift_val = None
        center_xyz = None
        sa_feat = None
        radius_adap = None
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            if use_center_shift==True or use_radius_update==True:
                new_xyz, new_points, idx, grouped_xyz, center_xyz, sa_feat, shift_val, radius_adap = combined_module(npoint, radius, nsample, xyz, points, use_xyz, bn_decay, is_training, scope, use_center_shift, use_radius_update) 
            else:
                new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)
           
        if sa_feat is not None:
            new_points = tf.concat([sa_feat, new_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx, shift_val, center_xyz, radius_adap

def get_fit_loss(end_points, alfa=1.0):
    losses = []
    for key in end_points:
        _,idx_knn = knn_point(1, end_points[key]['xyz'], end_points[key]['fps'] + end_points[key]['shift'])    
        xyz_nn = tf.squeeze(group_point(end_points[key]['xyz'], idx_knn), axis=-2)
        dist = tf.maximum(1e-8, tf.reduce_sum(tf.square(xyz_nn - (end_points[key]['fps'] + end_points[key]['shift'])), axis=-1))
        #dist = tf.sqrt(dist)/alfa
        losses += [tf.reduce_mean(dist)]
    return tf.add_n(losses)

def get_range_loss(end_points):
    losses = []
    for key in end_points:
        norm = tf.maximum(1e-8,tf.reduce_sum(end_points[key]['shift']**2, axis=-1))
        norm = tf.sqrt(norm)
        loss_shift = tf.maximum(0.0, norm - end_points[key]['rad'])
        losses += [tf.reduce_mean(loss_shift)]
    return tf.add_n(losses)

def get_rad_loss(end_points):
    losses = []
    for key in end_points:        
        loss_limit_below= tf.abs(tf.minimum(0.0, end_points[key]['rad_update']))
        loss_limit_upper= tf.maximum(0.0, end_points[key]['rad_update'] - 2*end_points[key]['rad'])
        losses += [tf.reduce_mean(loss_limit_below)+tf.reduce_mean(loss_limit_upper)]
    return tf.add_n(losses)

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
