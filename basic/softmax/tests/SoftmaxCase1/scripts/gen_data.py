import torch
import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    test_type = np.float16
    input_x = np.random.uniform(1, 10, [1,128,128]).astype(test_type)
    input_x.tofile("./input/input_x.bin")

    x_holder = tf.placeholder(input_x.dtype, shape=input_x.shape)

    re = tf.nn.softmax(input_x, -1)
    with tf.Session() as sess:
        result = sess.run(re, feed_dict={x_holder: input_x})
    golden = result.astype(test_type)
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
