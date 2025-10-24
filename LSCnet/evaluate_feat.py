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
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
#import modelnet_dataset
from sklearn import metrics
import modelnet_h5_dataset

DATASET = "Pepper"
METOT = "CSM_IV_1st_2nd_RUM_II_cum_1st_2nd"
log_file = 'log_'+METOT+'_'+DATASET
sys.path.append(os.path.join(BASE_DIR, log_file))
print(os.path.join(BASE_DIR, log_file))



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=DATASET, help='Plant name')
parser.add_argument('--gpu', type=int, default=1,help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default=os.path.join(log_file, 'model.ckpt'), help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default=os.path.join(log_file, 'dump'), help='dump folder path [dump]')
parser.add_argument('--num_votes', type=int, default=12, help='Aggregate classification scores from multiple rotations [default: 12]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

DUMP_DIR=FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

NUM_CLASSES = 2
DATA_PATH_TRAIN = '/home/data/cls/PLY_FILES_v2/'+FLAGS.dataset+'/train_files.txt'
DATA_PATH_TEST = '/home/data/cls/PLY_FILES_v2/'+FLAGS.dataset+'/test_files.txt'

SHAPE_NAMES = ['leaf', 'stem'] 

HOSTNAME = socket.gethostname()


TRAIN_DATASET = modelnet_h5_dataset.Dataset(DATA_PATH_TRAIN, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
TEST_DATASET = modelnet_h5_dataset.Dataset(DATA_PATH_TEST, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)


def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points_center,  end_points_rad = MODEL.get_model(pointclouds_pl, is_training_pl)
        MODEL.get_loss(pred, labels_pl, end_points_center,  end_points_rad)
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
    #log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}
    
    for i in range(10):
        LOG_DIR = os.path.join(DUMP_DIR, 'log_evaluate_' + str(i) + '.txt')
        PRED_DIR = os.path.join(DUMP_DIR, 'pred_' + str(i) + '.txt')
        GT_DIR = os.path.join(DUMP_DIR, 'gt_' + str(i) + '.txt')
        print "LOG:", LOG_DIR
        LOG_FOUT = open(LOG_DIR, 'w')
        LOG_PRED = open(PRED_DIR, 'w')
        LOG_GT = open(GT_DIR, 'w')
        LOG_FOUT.write(str(FLAGS)+'\n')
        eval_one_epoch(sess, ops, LOG_FOUT, LOG_PRED, LOG_GT, num_votes)
        LOG_FOUT.close()
        LOG_PRED.close()
        LOG_GT.close()

def eval_one_epoch(sess, ops, LOG_FOUT, LOG_PRED, LOG_GT, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3)) 
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    gt = []
    pred = []
    TEST_DATASET.reset()

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            np.random.shuffle(shuffled_indices)
            rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data[:, shuffled_indices, :],
                    vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val
        pred_val = np.argmax(batch_pred_sum, 1)

        for i in range(bsize):
            LOG_PRED.write(str(pred_val[i])+'\n')
            LOG_GT.write(str(batch_label[i])+'\n')
            
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
            pred.append(pred_val[i])
            gt.append(l)

    log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string(LOG_FOUT, 'eval accuracy: %.1f'% (np.round((total_correct / float(total_seen))*100,1)))
    log_string(LOG_FOUT, 'eval avg class acc: %.1f' % (np.round(np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))*100,1)) )
    log_string(LOG_FOUT, '%10s\t%10s\t%10s\t%10s\t%10s' % ("name", "accuracy", "precision", "recall", "F1"))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    precision = metrics.precision_score(gt, pred, average=None)
    recall = metrics.recall_score(gt, pred, average=None)
    f1 = metrics.f1_score(gt, pred, average=None)
    for i, name in enumerate(SHAPE_NAMES):
        log_string(LOG_FOUT, '%10s:\t%10.1f\t%10.1f\t%10.1f\t%10.1f' % (name, np.round(100*class_accuracies[i],1), np.round(100*precision[i],1), np.round(100*recall[i],1), np.round(100*f1[i],1)))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
   
