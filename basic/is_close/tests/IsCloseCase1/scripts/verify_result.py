import os
import sys
import numpy as np

loss = 1e-6 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.bool_) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.bool_) # 从bin文件读取预期运算结果
    difference, = np.where(real_result != golden)
  
    result_rtol = difference.size # 计算误差
   
    if result_rtol > real_result.size * loss : # 误差超出预期时返回打印错误，返回对比失败
        print("[ERROR] result error")
        return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
