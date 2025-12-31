import time
import numpy as np
import sys
from os import path
import os
import copy
import pickle
import shlex
import re
import pandas as pd
from datetime import datetime

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Real_Time_FJSP import RealTimeFJSPDual

if __name__ == "__main__":
    print("[*] Running evaluation...")
    
    # Parse command line arguments
    mode = sys.argv[2]
    assert mode in ['train', 'val'], f"Invalid mode: {mode}. Must be 'train' or 'val'"
    
    # Get dataset mode (train or test) - default to train for backward compatibility
    dataset_mode = sys.argv[4] if len(sys.argv) > 4 else "train"

    # Set up paths based on dataset mode
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset_mode == "test":
        # Test mode: use test dataset
        folder_path = os.path.join(base_dir, "test_data/test_jsp")
        real_folder_path = os.path.join(base_dir, "test_data/real_test_jsp")
        print(f"[*] Using TEST dataset: {folder_path}")
    else:
        # Train mode: use training dataset
        folder_path = os.path.join(base_dir, "test_data/jsp_cases")
        real_folder_path = os.path.join(base_dir, "test_data/real_jsp_cases")
        print(f"[*] Using TRAIN dataset: {folder_path}")
        
    values = []

    if mode == 'train':
        # Parse case numbers from command line
        case_numbers = [int(num) for num in shlex.split(sys.argv[3])]
        
        for idx, case_num in enumerate(case_numbers, 1):
            plan_filename = f"prob_{case_num:02d}.pkl"
            
            # Test dataset doesn't have _real suffix
            if dataset_mode == "test":
                real_filename = f"prob_{case_num:02d}.pkl"
            else:
                real_filename = f"prob_{case_num:02d}_real.pkl"

            plan_path = os.path.join(folder_path, plan_filename)
            real_path = os.path.join(real_folder_path, real_filename)

            start_time = time.time()

            # Load problem data
            with open(plan_path, 'rb') as f:
                plan_data = pickle.load(f)

            with open(real_path, 'rb') as f:
                real_data = pickle.load(f)
            
            # Create environment and run simulation
            env = RealTimeFJSPDual(plan_data, real_data)
            env.reset()
            
            while True:
                done = env.step()
                if done:
                    break
            
            # Record results
            makespan = env.current_t
            values.append(makespan)
            end_time = time.time()
            execution_time = end_time - start_time
            avg_ema = env.get_avg_ema()
            
            print(f"[*] Instance {idx}: makespan={makespan:.2f}, ema={avg_ema:.4f}, time={execution_time:.4f}s")

        # Print average makespan (primary objective)
        print("[*] Average:")
        print(np.mean(values))
