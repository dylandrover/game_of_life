import tensorflow as tf
import os
import sys
import time
import numpy as np

dim1 = int(sys.argv[1])
dim2 = int(sys.argv[2])
gen = int(sys.argv[3])

# Declare tf variables
#env  = tf.Variable(tf.abs(tf.round(tf.truncated_normal(shape=(dim1, dim2),
                                                       #mean=0, stddev=1))), name='state')

seed = tf.Variable([[0, 0, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 1, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0]], dtype=tf.float32)
neighbour_sum = tf.placeholder(tf.float32, shape=env.get_shape())

env = 
alive_dead    = tf.placeholder(tf.float32, shape=env.get_shape())
combin = tf.placeholder(tf.float32, shape=env.get_shape())
and_   = tf.placeholder(tf.float32, shape=env.get_shape())
less = tf.placeholder(tf.bool, shape=env.get_shape())
greater = tf.placeholder(tf.bool, shape=env.get_shape())

# Filter that sums neighbours 
sum_filt = tf.constant([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])

# Reshape to allow for conv2D
env = tf.reshape(env, [-1, dim1, dim2, 1])
sum_filt  = tf.reshape(sum_filt, [3, 3, 1, 1])

# Compile the Graph
neighbour_sum  = tf.nn.conv2d(env, sum_filt, [1, 1, 1, 1], padding="SAME", name='neighbours')
alive_dead     = tf.reshape(tf.add(env, 1), [dim1, dim2])
neighbour_sum = tf.reshape(neighbour_sum, [dim1, dim2])

combin = tf.multiply(neighbour_sum, alive_dead)
less = tf.cast(tf.greater(combin, tf.scalar_mul(6, tf.ones(shape=combin.get_shape()))), tf.int32)
greater = tf.cast(tf.greater_equal(combin, tf.scalar_mul(3, tf.ones(shape=combin.get_shape()))), tf.int32)
and_ = tf.cast(tf.add(greater,less, name='new_state'), tf.int32)
env = tf.reshape(env, [dim1, dim2])
env = and_
'''
life_summary = tf.summary.image('grid',and_)
merged = tf.summary.merge_all(life_summary)
'''

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

summary_writer = tf.summary.FileWriter('./tmp/logs/')

for i in range(gen):
    sess.run(init)
    #summary_writer.add_summary(merged, i)
    #env = and_
    print(env.eval(session=sess))
    print(and_.eval(session=sess))
    time.sleep(0.5)
    
