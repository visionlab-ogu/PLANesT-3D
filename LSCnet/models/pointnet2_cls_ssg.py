"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, get_range_loss, get_fit_loss, get_rad_loss

FLAG_CENTER_1 = True
FLAG_CENTER_2 = True
FLAG_RAD_1 = True
FLAG_RAD_2 = True

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points_center = {}
    end_points_rad = {}
    l0_xyz = point_cloud
    l0_points = None
  

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices, l1_shift, l1_fps, l1_rad = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True, use_xyz=True, use_center_shift=FLAG_CENTER_1, use_radius_update=FLAG_RAD_1)

    if FLAG_CENTER_1:    
        end_points_center[1] = {'xyz':l0_xyz, 'new_xyz':l1_xyz, 'shift':l1_shift, 'fps':l1_fps, 'rad':tf.constant(0.2)}
    if FLAG_RAD_1: 
        end_points_rad[1] = {'rad_update':l1_rad, 'rad':tf.constant(0.2)}

    l2_xyz, l2_points, l2_indices, l2_shift, l2_fps, l2_rad = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', use_xyz=True, use_center_shift=FLAG_CENTER_2, use_radius_update=FLAG_RAD_2)

    if FLAG_CENTER_2:    
        end_points_center[2] = {'xyz':l1_xyz, 'new_xyz':l2_xyz, 'shift':l2_shift, 'fps':l2_fps, 'rad':tf.constant(0.4)}
    if FLAG_RAD_2: 
        end_points_rad[2] = {'rad_update':l2_rad, 'rad':tf.constant(0.4)}

    l3_xyz, l3_points, l3_indices, l3_shift, l3_fps, l3_rad = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', use_xyz=True)
    
    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 2, activation_fn=None, scope='fc3')

    return net, end_points_center, end_points_rad


def get_loss(pred, label, end_points_center, end_points_rad):
    """ pred: B*NUM_CLASSES,
        label: B, """
        
    use_center_shift = FLAG_CENTER_1 or FLAG_CENTER_2
    use_radius_update = FLAG_RAD_1 or FLAG_RAD_2 
    
    weights_decay = 1e-3
    loss_shift_range_reg = 0.01
    loss_shift_fit_reg = 0.01
    loss_rad_limit_reg = 0.01
    
    """regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
    regularization_loss = weights_decay * tf.add_n(regularization_losses)"""
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)

    if use_center_shift and not use_radius_update:
        print "Loss: center shift"
        loss_shift_fit = loss_shift_fit_reg*get_fit_loss(end_points_center)
        loss_shift_range = loss_shift_range_reg*get_range_loss(end_points_center)
        tf.summary.scalar('shift range loss', loss_shift_range)
        tf.summary.scalar('shift fit loss', loss_shift_fit)      
        total_loss = classify_loss + loss_shift_fit + loss_shift_range 
        
    elif use_radius_update and not use_center_shift:
        print "Loss:  radius update"
        loss_rad_limit = loss_rad_limit_reg*get_rad_loss(end_points_rad)
        tf.summary.scalar('rad limit loss', loss_rad_limit)
        total_loss = classify_loss + loss_rad_limit

    elif use_radius_update and use_center_shift:
        print "Loss: center shift and radius update"
        loss_shift_fit = loss_shift_fit_reg*get_fit_loss(end_points_center)
        loss_shift_range = loss_shift_range_reg*get_range_loss(end_points_center)
        loss_rad_limit = loss_rad_limit_reg*get_rad_loss(end_points_rad)
        tf.summary.scalar('rad limit loss', loss_rad_limit)
        tf.summary.scalar('shift range loss', loss_shift_range)
        tf.summary.scalar('shift fit loss', loss_shift_fit)      
        total_loss = classify_loss + loss_shift_fit + loss_shift_range + loss_rad_limit
    
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', total_loss)
    return total_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
