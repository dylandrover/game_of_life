import tensorflow as tf
import os
import sys
import time
import numpy as np

dim1 = int(sys.argv[1])
dim2 = int(sys.argv[2])
gen = int(sys.argv[3])



# Declare tf variables
environment  = tf.Variable(tf.abs(tf.round(tf.truncated_normal(shape=(dim1, dim2), mean=0, stddev=0.5))), name='state')
sum_filter   = tf.constant([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])

environment = tf.reshape(environment, [-1, dim1, dim2, 1])
sum_filter  = tf.reshape(sum_filter, [3, 3, 1, 1])

neighbour_sum = tf.placeholder(tf.float32, shape=environment.get_shape())
alive_dead    = tf.placeholder(tf.float32, shape=environment.get_shape())
combin = tf.placeholder(tf.float32, shape=environment.get_shape())
and_   = tf.placeholder(tf.float32, shape=environment.get_shape())

less = tf.placeholder(tf.bool, shape=environment.get_shape())
greater = tf.placeholder(tf.bool, shape=environment.get_shape())


neighbour_sum  = tf.nn.conv2d(environment, sum_filter, [1, 1, 1, 1], padding="SAME", name='neighbours')
alive_dead     = tf.add(environment, 1)
combin = tf.multiply(neighbour_sum, alive_dead)
greater = tf.greater(combin, tf.scalar_mul(6, tf.ones(shape=combin.get_shape())))
less = tf.less(combin, tf.scalar_mul(3, tf.ones(shape=combin.get_shape())))
and_ = tf.cast(tf.logical_and(greater,less, name='new_state'), tf.float32)
life_summary = tf.summary.image('grid',and_)
merged = tf.summary.merge_all(life_summary)


init = tf.global_variables_initializer()

sess = tf.Session()

summary_writer = tf.summary.FileWriter('./tmp/logs/')

for i in range(gen):
    sess.run(init)
    summary_writer.add_summary(merged, i)
    environment = and_
    time.sleep(0.5)
    
