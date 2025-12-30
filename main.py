import hydra
import logging
import os
import sys
from pathlib import Path
import subprocess
from utils.utils import init_client

ROOT_DIR = os.getcwd()  # Get current working directory
@hydra.main(version_base=None, config_path="cfg", config_name="config")

def main(cfg):
    workspace_dir = Path.cwd()
    
    # Log workspace and configuration information
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")
    logging.info(f"Case num: {cfg.case_num}")

    # Initialize LLM client
    init_client(cfg)
    
    # Select and initialize algorithm
    if cfg.algorithm == "seevo":
        from seevo import SeEvo as Algorithm
    elif cfg.algorithm == "reevo":
        from reevo import ReEvo as Algorithm
    else:
        raise NotImplementedError(
            f"Algorithm '{cfg.algorithm}' is not implemented. "
            f"Available algorithms: 'seevo' (main), 'reevo' (baseline)"
        )

    # Run evolutionary algorithm
    algorithm = Algorithm(cfg, ROOT_DIR, [cfg.case_num])
    best_code_overall, best_code_path_overall = algorithm.evolve()
    
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
    
    # Save best code and run validation
    gpt_file_path = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py"
    with open(gpt_file_path, 'w') as file:
        file.write(best_code_overall + '\n')
    
    # Execute validation script
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script: {test_script}")
    
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run([sys.executable, test_script, "-1", ROOT_DIR, "val"], 
                      stdout=stdout, stderr=subprocess.STDOUT)
    
    logging.info(f"Validation script finished. Results saved in {test_script_stdout}")

if __name__ == "__main__":
    main()