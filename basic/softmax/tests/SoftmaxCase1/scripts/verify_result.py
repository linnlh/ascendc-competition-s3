import os
import sys
import numpy as np

loss = 1e-3 
minimum = 10e-10

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float16) 
    golden = np.fromfile(golden, dtype=np.float16) 
    result = np.abs(real_result - golden) 
    deno = np.maximum(np.abs(real_result), np.abs(golden))  
    result_atol = np.less_equal(result, loss) 
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss: 
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
