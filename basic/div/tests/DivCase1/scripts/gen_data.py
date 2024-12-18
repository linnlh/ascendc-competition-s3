#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [32, 64]
minval = 1
maxval = 10
dtype = tf.float16

def gen_golden_data_simple():

    input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_x.numpy().tofile("./input/input_x.bin")

    input_y = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_y.numpy().tofile("./input/input_y.bin")

    golden = tf.raw_ops.Div(x=input_x, y=input_y)
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

