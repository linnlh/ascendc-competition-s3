#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import tensorflow as tf
import os
    
np.random.seed(29)
tf.random.set_seed(29)
os.system("mkdir -p input")
os.system("mkdir -p output")

shape = [100]
minval = 0
maxval = 5
dtype = np.int8

def gen_golden_data_simple():

    input_x = np.random.randint(minval, maxval, size=shape).astype(np.int8)
    input_x.tofile("./input/input_x.bin")

    input_y = np.random.randint(minval, maxval, size=shape).astype(np.int8)
    input_y.tofile("./input/input_y.bin")

    golden = np.not_equal(input_x, input_y)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

