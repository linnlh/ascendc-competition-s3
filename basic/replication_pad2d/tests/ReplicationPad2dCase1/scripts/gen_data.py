import torch
import torch.nn as nn
import numpy as np
import os


def gen_golden_data_simple():

    test_type = np.float16
    paddings_type = np.int32
    input_x = np.random.uniform(-5, 5,[4,16,16] ).astype(test_type)
    input_paddings = [2,2,2,2]
    input_paddings = np.array(input_paddings).astype(paddings_type)
    flattened_array = tuple(input_paddings.flatten())
    mode = "replicate"
    res = torch.nn.functional.pad(torch.Tensor(input_x), flattened_array,mode)
    golden = res.numpy().astype(test_type)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_paddings.tofile("./input/paddings.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
