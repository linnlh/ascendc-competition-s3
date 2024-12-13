import numpy as np
import os
import torch


def get_golden(data, indices, updates, axis=0,reduction='none'):
    data = data.astype(np.float32)
    indices = indices.astype(np.int64)
    updates = updates.astype(np.float32)
    if reduction == "add":
        res = torch.scatter_add(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates))
    else:
        res = torch.scatter(torch.from_numpy(data), axis, torch.from_numpy(indices), torch.from_numpy(updates))
    res = res.numpy().astype(np.float16)
    return res


def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_data = np.random.uniform(-1, 1, [32, 64]).astype(np.float16)
    input_indices = np.random.uniform(0, 30, [32, 32]).astype(np.int32)
    input_updates = np.random.uniform(-10, 10, [32, 32]).astype(np.float16)
    input_data.tofile("./input/input_data.bin")
    input_indices.tofile("./input/input_indices.bin")
    input_updates.tofile("./input/input_updates.bin")
    axis = 1
    reduction = "add"
    golden = get_golden(input_data, input_indices, input_updates,axis=axis,reduction=reduction)

    print(golden)
    golden.tofile("./output/golden.bin")


def calc_expect_func(data, indices, updates, y, axis,reduction):
    input_data = data["value"]
    input_indices = indices["value"]
    input_updates = updates["value"] 
    res = get_golden(input_data, input_indices, input_updates,axis,reduction)

    return [res,]


if __name__ == "__main__":
    gen_golden_data_simple()
