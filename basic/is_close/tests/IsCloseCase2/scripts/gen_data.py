import numpy as np
import os
import torch


def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1 = np.random.uniform(-1, 1, [128]).astype(np.float32)
    tmp = np.random.uniform(-0.01, 0, [128]).astype(np.float32)
    input_x2 = input_x1 - tmp
    rtol = 0.005
    atol = 0.005
    equal_nan = False
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    res = torch.isclose(torch.tensor(input_x1), torch.tensor(input_x2), rtol=rtol, atol=atol, equal_nan=equal_nan)
    golden = res.numpy()
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
