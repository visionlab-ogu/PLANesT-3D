import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, query_ball_point_random,query_ball_point_pointwise, group_point

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:0'):
      #np.random.seed(13)
      points = tf.constant(np.random.random((50,128,16)).astype('float32'))
      xyz1 = tf.constant(np.random.random((50,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((50,64,3)).astype('float32'))
      radius = tf.constant((np.random.randint(0,3, size=(50,64,1))*0.22).astype('float32'))
      #radius = 0.3
      print radius
      nsample = 32
      idx, pts_cnt = query_ball_point_pointwise(radius, nsample, xyz1, xyz2)
      
      #grouped_points = group_point(points, idx)


    with self.test_session() as sess:
      print "---- Going to compute gradient error"
      data = sess.run(idx)
      print(data[0,0])
      """err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
      print err"""
      #self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
