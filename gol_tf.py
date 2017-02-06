import tensorflow as tf
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dim1 = int(sys.argv[1])
dim2 = int(sys.argv[2])
gen = int(sys.argv[3])
flag = None

if(len(sys.argv) == 5):
    flag = sys.argv[4]
    fig = plt.figure()

# Declare tf variables
env  = tf.Variable(tf.abs(tf.round(tf.truncated_normal(shape=(dim1, dim2),
                                                       mean=0, stddev=0.375))), name='state')
'''
Testing matrix for blinkers/stationary lifeforms
env = tf.Variable([[0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0]], dtype=tf.float32)
'''

neighbour_sum = tf.placeholder(tf.float32, shape=env.get_shape())

alive_dead    = tf.placeholder(tf.float32, shape=env.get_shape())
combin = tf.placeholder(tf.float32, shape=env.get_shape())
and_   = tf.placeholder(tf.float32, shape=env.get_shape())
less = tf.placeholder(tf.bool, shape=env.get_shape())
greater = tf.placeholder(tf.bool, shape=env.get_shape())

# Filter that sums neighbours 
sum_filt = tf.constant([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])

# Reshape to allow for conv2D
env_img = tf.reshape(env, [-1, dim1, dim2, 1])
sum_filt  = tf.reshape(sum_filt, [3, 3, 1, 1])

# Compile the Graph
neighbour_sum  = tf.nn.conv2d(env_img, sum_filt, [1, 1, 1, 1], padding="SAME", name='neighbours')
alive_dead     = tf.reshape(tf.subtract(tf.scalar_mul(2, env), tf.ones([dim1,dim2])), [dim1, dim2])
neighbour_sum = tf.reshape(neighbour_sum, [dim1, dim2])

combine = tf.multiply(neighbour_sum, alive_dead)

equals0  = tf.equal(combine, tf.scalar_mul(-3, tf.ones(shape=combine.get_shape())))
equals1 = tf.equal(combine, tf.scalar_mul(3, tf.ones(shape=combine.get_shape())))
equals2 = tf.equal(combine, tf.scalar_mul(2, tf.ones(shape=combine.get_shape())))

temp    = tf.logical_or(equals1,equals2)
new_val = tf.cast(tf.logical_or(temp, equals0, name='new_state'), tf.float32)
env_image = tf.reshape(new_val, [1, dim1, dim2, 1])

update = env.assign(new_val)

life_summary = tf.summary.image("board", env_image)
live_summary = tf.summary.scalar("sum", tf.reduce_sum(new_val))

# Merge summaries to write out to tensorboard
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./tb_summaries/')

if(flag == 'mpl'):
    ims = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(gen):
        img, summary = sess.run([update, merged])
        summary_writer.add_summary(summary, i)
        if(flag):
            img = plt.imshow(img, cmap='gray')
            ims.append([img])

# Creates animation of epochs and saves to gif file using imagemagick
if(flag == 'mpl'):
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=50)
    ani.save('./images/'+ time.strftime("%H:%M:%S") + '_life.gif', writer='imagemagick', fps=90)
    plt.show()

    

    
