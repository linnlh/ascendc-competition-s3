import torch
import torch.nn as nn
import numpy as np
import os    
def gen_golden_data_simple():    
    test_type = np.float32
    target_type = np.int32
    input_x = np.random.uniform(-5, 5,[1024, 1024] ).astype(test_type)
    input_target = np.random.uniform(0,1023,[1024] ).astype(target_type)
    input_weight= np.random.uniform(0,1,[1024] ).astype(test_type)
    reduction="mean";
    ignore_index=-100;
    res = torch.nn.functional.nll_loss(torch.Tensor(input_x), torch.Tensor(input_target).to(torch.long), weight=torch.Tensor(input_weight), size_average=None, 
                                       ignore_index=ignore_index, reduce=None, reduction=reduction)
    golden = res.numpy().astype(test_type)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_target.tofile("./input/target.bin")
    input_weight.tofile("./input/weight.bin")
    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    gen_golden_data_simple()