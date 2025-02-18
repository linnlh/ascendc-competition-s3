import numpy as np
import os



def gen_golden_data_simple():
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1 = np.random.uniform(-10, 10, [128,512]).astype(np.float32)
    input_x2 = np.random.uniform(-10, 10, [512,1024]).astype(np.float32)
    input_x3 = np.random.uniform(-10, 10, [1024]).astype(np.float32)

    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    input_x3.tofile("./input/input_x3.bin")
    golden = np.matmul(input_x1,input_x2) - input_x3

    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    gen_golden_data_simple()
