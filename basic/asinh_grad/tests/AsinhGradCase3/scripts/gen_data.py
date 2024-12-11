#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [1024, 1024]
minval = 1
maxval = 3
dtype = tf.float16

def gen_golden_data_simple():

    input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_y.numpy().tofile("./input/input_y.bin")

    input_dy = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_dy.numpy().tofile("./input/input_dy.bin")

    golden = input_dy / tf.raw_ops.Cosh(x=input_y)
    golden.numpy().tofile("./output/golden.bin")

    print("input_y is: ", input_y)
    print("input_dy is: ", input_dy)
    print("golden is: ", golden)

if __name__ == "__main__":
    gen_golden_data_simple()

