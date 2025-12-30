import time
import numpy as np
import sys
from os import path
import os
import copy
import pickle
# 添加 JSP_code 的父目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Real_Time_FJSP import RealTimeFJSPDual
import pandas as pd  
from datetime import datetime 
import shlex
import re

if __name__ == "__main__":
    print("[*] Running ...")
    # mood = 'val'
    mood = sys.argv[2]
    assert mood in ['train', 'val']

    # job_modified_expression = sys.argv[2]
    # machine_modified_expression = sys.argv[3]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "test_data/jsp_cases")
    real_folder_path = os.path.join(base_dir, "test_data/real_jsp_cases")
        
    value = [] 
    i = 0

    if mood == 'train':
        case_num = [int(num) for num in shlex.split(sys.argv[3])]
        for i in case_num:   
            plan_filename = f"prob_{i:02d}.pkl"
            real_filename = f"prob_{i:02d}_real.pkl"

            plan_path = os.path.join(folder_path, plan_filename)
            real_path = os.path.join(real_folder_path, real_filename)

            start_time = time.time()

            with open(plan_path, 'rb') as f:
                plan_data = pickle.load(f)

            with open(real_path, 'rb') as f:
                real_data = pickle.load(f) 
                            
            env = RealTimeFJSPDual(plan_data, real_data)

            for _ in range(1):
                env.reset() # 环境重置
                while True:
                    done = env.step()
                    if done:
                        break
                    
            value.append(env.current_t)
            end_time = time.time()  
            execution_time = end_time - start_time  
            i = i + 1
            print(f"[*] Instance {i}: {env.current_t} {execution_time} 秒")

        print("[*] Average:")
        print(np.mean(value))

    # else:
    #     result = []
    #     for file in files:    
    #         file_path = os.path.join(folder_path, file)  # 获取文件的完整路径  
    #         with open(file_path, 'r') as f:  # 打开文件  
    #             data = f.readlines()  # 读取文件内容  
    #             dimensions = list(map(int, data[0].split()))  
    #             num_jobs = dimensions[0]  
    #             num_ops = dimensions[1]  

    #             machine_matrix = []  
    #             time_matrix = []  

    #             for line in data[1:]:  
    #                 job_data = list(map(int, line.split()))  
    #                 machines = job_data[::2]  
    #                 times = job_data[1::2]  
    #                 machine_matrix.append(machines)  
    #                 time_matrix.append(times)  

    #             print(f"[*] Dataset loaded: {file} {num_jobs} {num_ops} instances.")

    #             pts = np.array(time_matrix) # 提取对应的数据集
    #             ops = np.array(machine_matrix)

    #             start_time = time.time() 
    #             env = Real_Time_FJSP_mach_order_env(pts=pts, ops=ops, Gantt=True)
    #             env.num_job = len(ops)    # 更新新的工件工序数量，加工时间
    #             env.num_op = len(ops[0]) 
    #             env.pts = pts
    #             env.ops = ops
                
                
    #             for _ in range(1):
    #                 env.reset() # 环境重置
    #                 while True:
    #                     done = env.step()
    #                     if done:
    #                         break
    #             value.append(env.current_time)
    #             end_time = time.time()  
    #             execution_time = end_time - start_time  
    #             i = i + 1
    #             print(f"[*] Instance {i}: {env.current_time} {execution_time} 秒")
    #             result.append([file, env.current_time])

    #     print("[*] Average:")
    #     print(np.mean(value))
    #     # 将结果列表转换成DataFrame  
    #     df = pd.DataFrame(result, columns=['File Name', 'Makespan'])  
        
    #     # 按照"File Name"列的值进行排序  
    #     df_sorted = df.sort_values(by='File Name')  
        
    #     # 将排序后的结果保存至新的CSV文件  
    #     now = datetime.now() 
    #     file_name = now.strftime("%Y%m%d_%H%M_" + "result.csv") 
    #     df_sorted.to_csv(file_name, index=False)
         
