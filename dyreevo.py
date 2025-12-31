from openai import OpenAI
import logging
import subprocess
import numpy as np
import os
from time import time
os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from utils.utils import *
import random

python_executable = '/home/sshj/miniconda3/envs/chatgpt/bin/python'

class ReEvo:
    def __init__(self, cfg, root_dir, case_num) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate  # 变异概率
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float("inf")
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.case_num = case_num

        self.init_prompt()


    def init_prompt(self) -> None: 
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description  # 问题描述
        self.problem_size = self.cfg.problem.problem_size 
        self.func_name = self.cfg.problem.func_name  # 启发式
        self.obj_type = self.cfg.problem.obj_type  # 优化目标类型，min最小目标
        self.problem_type = self.cfg.problem.problem_type  # 目标类型
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"  # 输出文件.py
        self.input_dir = f"{self.root_dir}/data"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')  # 需要修改这一部分

        case_num = 0
        population= []
        for folder in os.listdir(self.input_dir):
            case_path = os.path.join(self.input_dir, folder)
            if os.path.isdir(case_path):
                for file in os.listdir(case_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(case_path, file)
                        code = extract_code_from_generator(file_to_string(file_path))
                        seed_ind = {
                            "stdout_filepath": f"problem_iter{self.iteration}_stdout{case_num}.txt",
                            "code_path": f"problem_iter{self.iteration}_code{case_num}.py",
                            "code": code,
                            "response_id": 0,
                        } #  这个是一个字典
                        case_num += 1
                        population.append(seed_ind)
        
        population = self.evaluate_population(population, self.case_num)
        self.population = population
        self.update_iter()

        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')  #函数描述，这一部分需要修改
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_reflection_str = self.external_knowledge  # 长期反射提示词
        else:
            self.external_knowledge = ""
        
        
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # long-term reflection
        self.user_reflector_ise_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_ise.txt')
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.individual_self_evolution_prompt = file_to_string(f'{self.prompt_dir}/common/Individual_self_evolution.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
            )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration
        self.print_individual_self_evolution_prompt = True
        self.print_individual_self_evolution_reflection_prompt = True

    def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name[:-4] + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def evaluate_population(self, population: list[dict], case_num: list) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)): # 查看多少个response_id
            # self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:  # 如果没有程序则直接报错
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id, case_num)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() # 把所有验证集测试结果进行测试
            traceback_msg = filter_traceback(stdout_str) # 检查是否有问题
            
            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2]) # 倒数第二个数据是均值
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    # def _run_code(self, individual: dict, response_id, case_num: list) -> subprocess.Popen:
    #     """
    #     Write code into a file and run eval script.
    #     """
    #     logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
    #     with open(self.output_file, 'w') as file:
    #         file.writelines(individual["code"] + '\n')  # 将程序进行保存
    #         file.flush()  # 确保缓冲区内容写入磁盘
    #         os.fsync(file.fileno())  # 强制操作系统将缓冲区内容写入磁盘

    #     # Execute the python file with flags
    #     case_num_str = ' '.join(map(str, case_num))
    #     with open(individual["stdout_filepath"], 'w') as f:
    #         eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
    #         process = subprocess.Popen([python_executable, '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train", case_num_str],
    #                                     stdout=f, stderr=f)

    #     block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
    #     return process

    def _run_code(self, individual: dict, response_id, case_num: list) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')  # 将程序进行保存
            file.flush()  # 确保缓冲区内容写入磁盘
            os.fsync(file.fileno())  # 强制操作系统将缓冲区内容写入磁盘

        # Execute the python file with flags
        case_num_str = ' '.join(map(str, case_num))
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
            process = subprocess.Popen([python_executable, '-u', eval_file_path, self.root_dir, "train", case_num_str],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process
    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))  # 找到最小值和最小值对应的位置
        
        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def population_inter_envoltion_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            print(ind1["code"], ind2["code"])
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code
    
    def individual_self_evolution_reflection_prompt(self, ind: dict, ind1: dict, ind2: dict, older_prompt: str) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] < ind2["obj"]:
            better_ind = ind1
        elif ind1["obj"] > ind2["obj"]:
            better_ind = ind2

        if better_ind["obj"] < ind["obj"]:
            result = "worse"
        elif better_ind["obj"] == ind["obj"]:
            result = "same"
        else:
            result = "better"

        better_code = filter_code(better_ind["code"])
        new_code = filter_code(ind["code"])

        system = self.system_reflector_prompt
        user = self.user_reflector_ise_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            result = result,
            first_version_code = better_code,
            second_version_code = new_code,
            hint = older_prompt
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_individual_self_evolution_reflection_prompt:
            logging.info("Individual Self Evolution Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_individual_self_evolution_reflection_prompt = False
        return message, better_code, new_code  

    def population_inter_envoltion_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.population_inter_envoltion_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)  # 短期反射
            better_code_lst.append(better_code) # 长期反射
        
        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        return response_lst, worse_code_lst, better_code_lst
    
    def individual_self_evolution_reflection(self, population: list[dict], population_inter_envoltion_reflection_tuple: list[dict], selected_population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        older_code_lst = []
        now_code_list = []
        for i in range(0, len(selected_population), 2):
            # Select two individuals
            new_individual = population[i//2]
            parent_1 = selected_population[i]
            parent_2 = selected_population[i+1]
            older_prompt = population_inter_envoltion_reflection_tuple[i//2]
            
            # Short-term reflection
            messages, better_code, new_code = self.individual_self_evolution_reflection_prompt(new_individual, parent_1, parent_2, older_prompt)
            messages_lst.append(messages)
            older_code_lst.append(better_code) 
            now_code_list.append(new_code) 
        
        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        return response_lst, older_code_lst, now_code_list

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_reflection = self.long_term_reflection_str,
            new_reflection = "\n".join(short_term_reflections),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        self.long_term_reflection_str = multi_chat_completion([messages], 1, self.cfg.model, self.cfg.temperature)[0]
        
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')


    def population_inter_envoltion(self, population_inter_envoltion_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = population_inter_envoltion_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def individual_self_evolution(self, population_inter_envoltion_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, better_code_lst, new_code_lst = population_inter_envoltion_reflection_tuple
        messages_lst = []
        for reflection, older_code, new_code in zip(reflection_content_lst, better_code_lst, new_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.individual_self_evolution_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                older_code = older_code,
                new_code = new_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_individual_self_evolution_prompt:
                logging.info("Individual Self Evolution Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_individual_self_evolution_prompt = False
        
        # Asynchronously generate responses
        response_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1) 
        user = self.mutataion_prompt.format(
            user_generator = self.user_generator_prompt,
            reflection = self.long_term_reflection_str + self.external_knowledge,
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate), self.cfg.model, self.cfg.temperature)
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        return population


    def evolve(self):
        # If all individuals are invalid, stop
        if all([not individual["exec_success"] for individual in self.population]):
            raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
        # Select
        population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population # add elitist to population for selection，把最优个体返回至种群中
        selected_population = self.random_select(population_to_select)
        if selected_population is None:
            raise RuntimeError("Selection failed. Please check the population.")
        
        # Population inter envoltion reflection
        population_inter_envoltion_reflection_tuple = self.population_inter_envoltion_reflection(selected_population) # (response_lst, worse_code_lst, better_code_lst)
        population_inter_envoltion_population = self.population_inter_envoltion(population_inter_envoltion_reflection_tuple)
        # Evaluate
        self.population = self.evaluate_population(population_inter_envoltion_population, self.case_num)
        self.update_iter()

        # Individual self evolution reflection
        individual_self_evolution_reflection_tuple = self.individual_self_evolution_reflection(self.population,  population_inter_envoltion_reflection_tuple[0], selected_population)
        individual_self_evolution_population = self.individual_self_evolution(individual_self_evolution_reflection_tuple)
        self.population = self.evaluate_population(individual_self_evolution_population, self.case_num)
        self.update_iter()
        
        self.function_evals += 1
        
        return self.best_code_overall, self.best_code_path_overall
