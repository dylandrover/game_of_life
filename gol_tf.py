import tensorflow as tf
import os
import sys
import numpy as np

dim = sys.argv[1]
gen = sys.argv[2]

# Declare tf variables
grid = tf.Variable(tf.round(tf.random_normal(shape=(dim, dim), mean=0, stddev=1)))


# Dictates the state of each value in the main tf grid
def rules(grid):
    # A live cell with fewer than two live neighbours dies

    # A live cell with two or three live neighbours lives

    # A live cell with more than three live neighbours dies

    # A dead ell with exactly three live neighbours becomes a live cell

with 
for i in gen:
    
