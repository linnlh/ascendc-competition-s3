#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [4, 5, 6, 32, 32]
minval = -100
maxval = -1
dtype = tf.float16

def gen_golden_data_simple():

    input_x = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    input_x.numpy().tofile("./input/input_x.bin")

    golden = tf.raw_ops.Asinh(x=input_x)
    golden.numpy().tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

