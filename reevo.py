from openai import OpenAI
import logging
import subprocess
import numpy as np
import os
import sys
from time import time

os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from utils.utils import *
import random

class ReEvo:
    """Reflective Evolution (ReEvo) Algorithm.
    
    An LLM-based evolutionary algorithm that uses reflection mechanisms
    to evolve heuristics for optimization problems.
    """
    
    def __init__(self, cfg, root_dir, case_num) -> None:
        """Initialize ReEvo algorithm.
        
        Args:
            cfg: Configuration object containing algorithm parameters
            root_dir: Root directory of the project
            case_num: List of case numbers for evaluation
        """
        self.cfg = cfg
        self.root_dir = root_dir
        self.case_num = case_num
        self.mode = cfg.mode  # train or test
        
        # Algorithm parameters
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        
        # Population tracking
        self.elitist = None
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.current_ema = 0.0  # Track current solution's EMA
        
        # Reflection mechanism
        self.long_term_reflection_str = ""
        
        # Performance tracking for intelligent evolution guidance (only in train mode)
        if self.mode == "train":
            self.improvement_history = []  # Track improvement trends
            self.best_obj_history = []      # Track best objective over iterations
        
        # Initialize prompts and population
        self.init_prompt()
        self.init_population()


    def init_prompt(self) -> None:
        """Initialize prompt templates and problem-specific settings."""
        # Problem configuration
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type  # "min" or "max"
        self.problem_type = self.cfg.problem.problem_type
        
        # Log problem information
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        # Set up file paths
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # Load problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        
        # Load external knowledge if available
        external_knowledge_path = f'{problem_prompt_path}/external_knowledge.txt'
        if os.path.exists(external_knowledge_path):
            self.external_knowledge = file_to_string(external_knowledge_path)
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Load common prompt templates
        common_prompt_dir = f'{self.prompt_dir}/common'
        self.system_generator_prompt = file_to_string(f'{common_prompt_dir}/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{common_prompt_dir}/system_reflector.txt')
        
        # Short-term reflection prompt
        if self.problem_type != "black_box":
            self.user_reflector_st_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_st.txt')
        else:
            self.user_reflector_st_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_st_black_box.txt')
        
        # Long-term reflection and evolution prompts
        self.user_reflector_lt_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_lt.txt')
        self.user_reflector_ise_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_ise.txt')
        self.crossover_prompt = file_to_string(f'{common_prompt_dir}/crossover.txt')
        self.individual_self_evolution_prompt = file_to_string(f'{common_prompt_dir}/Individual_self_evolution.txt')
        self.mutation_prompt = file_to_string(f'{common_prompt_dir}/mutation.txt')
        # Format user prompts with problem-specific information
        self.user_generator_prompt = file_to_string(f'{common_prompt_dir}/user_generator.txt').format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f'{common_prompt_dir}/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flags to control prompt logging (print only once for the first iteration)
        self.print_crossover_prompt = True
        self.print_mutate_prompt = True
        self.print_short_term_reflection_prompt = True
        self.print_long_term_reflection_prompt = True
        self.print_individual_self_evolution_prompt = True
        self.print_individual_self_evolution_reflection_prompt = True



    def init_population(self) -> None:
        """Initialize population with seed function and LLM-generated individuals (train mode)
        or load trained rules from data directory (test mode)."""
        
        if self.mode == "test":
            # Test mode: Load trained rules from data_dir
            logging.info(f"Test mode: Loading trained rules from {self.cfg.data_dir}...")
            data_dir = os.path.join(self.root_dir, self.cfg.data_dir)
            
            if not os.path.exists(data_dir):
                raise RuntimeError(f"Data directory not found: {data_dir}")
            
            population = []
            response_id = 0
            
            # Load all .txt files from subdirectories
            for folder in sorted(os.listdir(data_dir)):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(folder_path, file)
                            code = extract_code_from_generator(file_to_string(file_path))
                            
                            individual = {
                                "stdout_filepath": f"problem_iter{self.iteration}_stdout{response_id}.txt",
                                "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
                                "code": code,
                                "response_id": response_id,
                            }
                            population.append(individual)
                            response_id += 1
            
            if len(population) == 0:
                raise RuntimeError(f"No trained rules found in {data_dir}")
            
            logging.info(f"Loaded {len(population)} trained rules")
            
            # Evaluate loaded population
            self.population = self.evaluate_population(population, self.case_num)
            self.update_iter()
            
        else:
            # Train mode: Normal initialization with seed and LLM generation
            logging.info("Train mode: Evaluating seed function...")
            code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
            logging.info("Seed function code: \n" + code)
            
            seed_ind = {
                "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
                "code_path": f"problem_iter{self.iteration}_code0.py",
                "code": code,
                "response_id": 0,
            }
            self.seed_ind = seed_ind
            self.population = self.evaluate_population([self.seed_ind], self.case_num)

            # Validate seed function
            if not self.seed_ind["exec_success"]:
                raise RuntimeError(
                    f"Seed function is invalid. Please check the stdout file in {os.getcwd()}."
                )

            self.update_iter()
            
            # Generate initial population using LLM
            system = self.system_generator_prompt
            user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

            # Increase temperature for diverse initial population
            responses = multi_chat_completion(
                [messages], 
                self.cfg.init_pop_size, 
                self.cfg.model, 
                self.cfg.temperature + 0.3
            )
            population = [self.response_to_individual(response, response_id) 
                         for response_id, response in enumerate(responses)]

            # Evaluate generated population
            population = self.evaluate_population(population, self.case_num)

            # Update iteration and population
            self.population = population
            self.update_iter()

    # def post_thought(self, code, algorithm):

    #     prompt_content = self.get_prompt_refine(code, algorithm)
        
    #     return prompt_content

    # def get_prompt_refine(self, code, algorithm):

    #     prompt_content = "Solving Job Shop Scheduling Problem (JSP) with constructive heuristics. JSP requires achieving the minimum makespan through effective production arrangement of workpieces and operations." + "\n"
    #     prompt_content += "The following describes the heuristic algorithm for the problem and the code with function name '" + self.func_name + "'.\n"
    #     prompt_content += "\nAlgorithm Design:\n" + algorithm
    #     if code == None:
    #         prompt_content += "\n\nCode:\n" + self.best_code_overall # 这帧数据是准备放弃的，没有意义我选择最好的个体，该数据无法正常运行肯定会报错
    #     else:            
    #         prompt_content += "\n\nCode:\n" + code

    #     prompt_content += "\n\nPlease summarize the algorithm's idea in **no more than 30 words** without any code or excessive detail. Be brief and clear."

    #     return prompt_content

    
    def response_to_individual(self, response: str, response_id: int, file_name: str = None) -> dict:
        """Convert LLM response to an individual dictionary.
        
        Args:
            response: LLM-generated response containing code
            response_id: Unique identifier for this response
            file_name: Optional custom filename for saving response
            
        Returns:
            Dictionary representing an individual with code and metadata
        """
        # Save response to file
        if file_name is None:
            file_name = f"problem_iter{self.iteration}_response{response_id}.txt"
        else:
            file_name = file_name + ".txt"
            
        with open(file_name, 'w') as file:
            file.write(response + '\n')

        # Extract code from response
        code = extract_code_from_generator(response)

        # Determine stdout filepath
        if file_name.endswith("_response" + str(response_id) + ".txt"):
            std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt"
        else:
            std_out_filepath = file_name[:-4] + "_stdout.txt"
        
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


    def evaluate_population(self, population: list[dict], case_num: list) -> list[dict]:
        """Evaluate population by running code in parallel and computing objective values.
        
        Args:
            population: List of individual dictionaries containing code
            case_num: List of test case numbers to evaluate on
            
        Returns:
            Updated population with objective values and execution status
        """
        inner_runs = []
        
        # Execute code for each individual
        for response_id in range(len(population)):
            # Skip if response contains no valid code
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id, case_num)
                inner_runs.append(process)
            except Exception as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_runs.append(None)
        
        # Collect results and update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:
                continue
                
            # Wait for code execution to finish
            try:
                inner_run.communicate(timeout=self.cfg.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Timeout for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            
            # Read execution output
            with open(stdout_filepath, 'r') as f:
                stdout_str = f.read()
            
            # Check for execution errors
            traceback_msg = filter_traceback(stdout_str)
            
            # Parse objective value if no errors
            if traceback_msg == '':
                try:
                    # Extract objective value (second-to-last line is the mean value)
                    obj_value = float(stdout_str.split('\n')[-2])
                    # Negate for maximization problems
                    individual["obj"] = obj_value if self.obj_type == "min" else -obj_value
                    individual["exec_success"] = True
                    
                    # Extract EMA value from output lines
                    import re
                    ema_matches = re.findall(r'ema=([0-9.]+)', stdout_str)
                    if ema_matches:
                        individual["ema"] = np.mean([float(x) for x in ema_matches])
                    else:
                        individual["ema"] = 0.0
                except Exception:
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], "Invalid stdout / objective value!"
                    )
            else:
                # Mark individual as invalid if execution failed
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            logging.info(
                f"Iteration {self.iteration}, response_id {response_id}: "
                f"Objective value: {individual['obj']}"
            )
        return population

    def _run_code(self, individual: dict, response_id: int, case_num: list) -> subprocess.Popen:
        """Write code to file and execute evaluation script.
        
        Args:
            individual: Individual dictionary containing code to execute
            response_id: Unique identifier for this individual
            case_num: List of test case numbers
            
        Returns:
            Subprocess handle for the running evaluation
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        # Write code to output file
        with open(self.output_file, 'w') as file:
            file.write(individual["code"] + '\n')
            file.flush()  # Ensure buffer is written to disk
            os.fsync(file.fileno())  # Force OS to write buffer to disk

        # Execute evaluation script
        case_num_str = ' '.join(map(str, case_num))
        eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
        
        # Pass framework mode to eval.py for dataset selection
        dataset_mode = "test" if self.mode == "test" else "train"
        
        with open(individual["stdout_filepath"], 'w') as f:
            process = subprocess.Popen(
                [sys.executable, '-u', eval_file_path, self.root_dir, "train", case_num_str, dataset_mode],
                stdout=f, 
                stderr=f
            )

        # Wait until evaluation starts
        block_until_running(
            individual["stdout_filepath"], 
            log_status=True, 
            iter_num=self.iteration, 
            response_id=response_id
        )
        return process
    
    def update_iter(self) -> None:
        """Update statistics and advance iteration counter."""
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj = min(objs)
        best_sample_idx = np.argmin(np.array(objs))
        
        # Track improvement (train mode only)
        if self.mode == "train":
            previous_best = self.best_obj_overall
            
            # Update best overall solution
            if self.best_obj_overall is None or best_obj < self.best_obj_overall:
                improvement = 0 if previous_best is None else (previous_best - best_obj)
                self.improvement_history.append(improvement)
                self.best_obj_overall = best_obj
                self.best_code_overall = population[best_sample_idx]["code"]
                self.best_code_path_overall = population[best_sample_idx]["code_path"]
                logging.info(f"Iteration {self.iteration}: Improvement = {improvement:.4f}")
            else:
                self.improvement_history.append(0)
            
            # Track best objective history
            self.best_obj_history.append(best_obj)
        else:
            # Test mode: just update best solution without tracking history
            if self.best_obj_overall is None or best_obj < self.best_obj_overall:
                self.best_obj_overall = best_obj
                self.best_code_overall = population[best_sample_idx]["code"]
                self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # Update elitist individual
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: New Elitist: {self.elitist['obj']}")
        
        # Log iteration summary
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """Rank-based selection with probability proportional to rank.
        
        Args:
            population: List of individuals to select from
            
        Returns:
            Selected population or None if selection fails
        """
        # Filter valid individuals
        if self.problem_type == "black_box":
            population = [
                ind for ind in population 
                if ind["exec_success"] and ind["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [ind for ind in population if ind["exec_success"]]
            
        if len(population) < 2:
            return None
            
        # Sort population by objective value and compute selection probabilities
        population = sorted(population, key=lambda x: x["obj"])
        ranks = list(range(len(population)))
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        probs = [prob / sum(probs) for prob in probs]  # Normalize
        
        # Select parents
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            # Ensure parents have different objective values
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """Random selection with equal probability for all individuals.
        
        Args:
            population: List of individuals to select from
            
        Returns:
            Selected population or None if selection fails
        """
        # Filter valid individuals
        if self.problem_type == "black_box":
            population = [
                ind for ind in population 
                if ind["exec_success"] and ind["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [ind for ind in population if ind["exec_success"]]
            
        if len(population) < 2:
            return None
            
        # Randomly select parent pairs
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # Ensure parents have different objective values (not identical)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def population_inter_envoltion_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """Generate short-term reflection prompt by comparing two individuals.
        
        Args:
            ind1: First individual
            ind2: Second individual
            
        Returns:
            Tuple of (messages, worse_code, better_code)
        """
        if ind1["obj"] == ind2["obj"]:
            raise ValueError(
                f"Two individuals to crossover have the same objective value: {ind1['obj']}"
            )
        
        # Determine which individual is better
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        
        # Calculate performance gap and provide context
        performance_gap = abs(better_ind["obj"] - worse_ind["obj"])
        gap_percentage = (performance_gap / abs(worse_ind["obj"]) * 100) if worse_ind["obj"] != 0 else 0
        
        # Add EMA adaptation context
        worse_ema = worse_ind.get("ema", 0.0)
        better_ema = better_ind.get("ema", 0.0)
        
        ema_guidance = ""
        if worse_ema > 0.01 or better_ema > 0.01:
            ema_diff = abs(worse_ema - better_ema)
            if better_ema < worse_ema:
                ema_guidance = (
                    f"\n\n**Uncertainty Adaptation Analysis:**\n"
                    f"- Worse code EMA: {worse_ema:.4f} (higher deviation from planned times)\n"
                    f"- Better code EMA: {better_ema:.4f} (better adaptation to uncertainty)\n"
                    f"- The better code adapts {ema_diff:.4f} better to fuzzy processing times.\n"
                    f"- **Key insight**: Better code likely uses EMA-adjusted priorities (e.g., pt*(1+ema)) "
                    f"to anticipate delays and make more robust decisions.\n"
                )
            elif ema_diff > 0.02:
                ema_guidance = (
                    f"\n\n**Uncertainty Adaptation Note:**\n"
                    f"- Both codes show different EMA patterns: worse={worse_ema:.4f}, better={better_ema:.4f}\n"
                    f"- Consider how the better code handles time uncertainties differently.\n"
                )
        
        performance_context = (
            f"\n\n**Performance Comparison:**\n"
            f"- Worse code objective: {worse_ind['obj']:.4f}\n"
            f"- Better code objective: {better_ind['obj']:.4f}\n"
            f"- Performance gap: {performance_gap:.4f} ({gap_percentage:.2f}% improvement)\n"
            f"- This gap is {'significant' if gap_percentage > 5 else 'moderate' if gap_percentage > 1 else 'small'}. "
            f"Focus on identifying {'key algorithmic differences' if gap_percentage > 5 else 'subtle optimizations'}.\n"
        ) + ema_guidance
        
        # Create reflection prompt
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
        ) + performance_context
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code
    
    def individual_self_evolution_reflection_prompt(self, ind: dict, ind1: dict, ind2: dict, older_prompt: str) -> tuple[list[dict], str, str]:
        """Generate reflection prompt for individual self-evolution.
        
        Args:
            ind: Current individual to evolve
            ind1: First parent individual
            ind2: Second parent individual
            older_prompt: Previous reflection/hint
            
        Returns:
            Tuple of (messages, better_code, new_code)
        """
        # Determine which parent is better
        better_ind = ind1 if ind1["obj"] < ind2["obj"] else ind2

        # Compare current individual with better parent
        if better_ind["obj"] < ind["obj"]:
            result = "worse"
        elif better_ind["obj"] == ind["obj"]:
            result = "same"
        else:
            result = "better"

        better_code = filter_code(better_ind["code"])
        new_code = filter_code(ind["code"])

        # Create reflection prompt
        system = self.system_reflector_prompt
        user = self.user_reflector_ise_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            result=result,
            first_version_code=better_code,
            second_version_code=new_code,
            hint=older_prompt
        )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        if self.print_individual_self_evolution_reflection_prompt:
            logging.info("Individual Self Evolution Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_individual_self_evolution_reflection_prompt = False
        return message, better_code, new_code  

    def population_inter_envoltion_reflection(self, population: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """Perform short-term reflection for population inter-evolution.
        
        Args:
            population: List of selected individuals (must be even number)
            
        Returns:
            Tuple of (reflection_responses, worse_code_list, better_code_list)
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        
        # Generate reflection prompts for each parent pair
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]
            
            # Generate short-term reflection
            messages, worse_code, better_code = self.population_inter_envoltion_reflection_prompt(
                parent_1, parent_2
            )
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Generate LLM responses asynchronously
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        return response_lst, worse_code_lst, better_code_lst
    
    def individual_self_evolution_reflection(self, population: list[dict], population_inter_envoltion_reflection_tuple: list[str], selected_population: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """Perform reflection for individual self-evolution.
        
        Args:
            population: Current population after inter-evolution
            population_inter_envoltion_reflection_tuple: Previous reflection responses
            selected_population: Original selected parent population
            
        Returns:
            Tuple of (reflection_responses, older_code_list, new_code_list)
        """
        messages_lst = []
        older_code_lst = []
        now_code_list = []
        
        # Generate reflection prompts for each individual
        for i in range(0, len(selected_population), 2):
            new_individual = population[i // 2]
            parent_1 = selected_population[i]
            parent_2 = selected_population[i + 1]
            older_prompt = population_inter_envoltion_reflection_tuple[i // 2]
            
            # Generate self-evolution reflection
            messages, better_code, new_code = self.individual_self_evolution_reflection_prompt(
                new_individual, parent_1, parent_2, older_prompt
            )
            messages_lst.append(messages)
            older_code_lst.append(better_code)
            now_code_list.append(new_code)
        
        # Generate LLM responses asynchronously
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        return response_lst, older_code_lst, now_code_list

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """Perform long-term reflection by aggregating short-term insights.
        
        Args:
            short_term_reflections: List of short-term reflection responses
        """
        # Create long-term reflection prompt
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        # Generate long-term reflection
        self.long_term_reflection_str = multi_chat_completion(
            [messages], 1, self.cfg.model, self.cfg.temperature
        )[0]
        
        # Save reflections to files
        short_term_file = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(short_term_file, 'w') as file:
            file.write("\n".join(short_term_reflections) + '\n')
        
        long_term_file = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(long_term_file, 'w') as file:
            file.write(self.long_term_reflection_str + '\n')


    def population_inter_envoltion(self, population_inter_envoltion_reflection_tuple: tuple[list[str], list[str], list[str]]) -> list[dict]:
        """Perform population inter-evolution (crossover) based on reflections.
        
        Args:
            population_inter_envoltion_reflection_tuple: Tuple of (reflections, worse_codes, better_codes)
            
        Returns:
            List of new individuals generated from crossover
        """
        reflection_content_lst, worse_code_lst, better_code_lst = population_inter_envoltion_reflection_tuple
        messages_lst = []
        
        # Generate crossover prompts for each pair
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Log prompt for the first iteration only
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Generate new individuals asynchronously
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        crossed_population = [
            self.response_to_individual(response, response_id) 
            for response_id, response in enumerate(response_lst)
        ]

        assert len(crossed_population) == self.cfg.pop_size, \
            f"Expected {self.cfg.pop_size} individuals, got {len(crossed_population)}"
        return crossed_population


    def individual_self_evolution(self, individual_self_evolution_reflection_tuple: tuple[list[str], list[str], list[str]]) -> list[dict]:
        """Perform individual self-evolution based on reflections.
        
        Args:
            individual_self_evolution_reflection_tuple: Tuple of (reflections, older_codes, new_codes)
            
        Returns:
            List of evolved individuals
        """
        reflection_content_lst, better_code_lst, new_code_lst = individual_self_evolution_reflection_tuple
        messages_lst = []
        
        # Generate self-evolution prompts for each individual
        for reflection, older_code, new_code in zip(reflection_content_lst, better_code_lst, new_code_lst):
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.individual_self_evolution_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                older_code=older_code,
                new_code=new_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Log prompt for the first iteration only
            if self.print_individual_self_evolution_prompt:
                logging.info("Individual Self Evolution Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_individual_self_evolution_prompt = False
        
        # Generate evolved individuals asynchronously
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        evolved_population = [
            self.response_to_individual(response, response_id) 
            for response_id, response in enumerate(response_lst)
        ]

        assert len(evolved_population) == self.cfg.pop_size, \
            f"Expected {self.cfg.pop_size} individuals, got {len(evolved_population)}"
        return evolved_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation.
        
        Mutates the best individual to generate new individuals based on
        long-term reflection and external knowledge.
        
        Returns:
            List of mutated individuals
        """
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=self.long_term_reflection_str + self.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(self.elitist["code"]),
            func_name=self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        
        # Generate mutated individuals
        num_mutations = int(self.cfg.pop_size * self.mutation_rate)
        responses = multi_chat_completion(
            [messages], num_mutations, self.cfg.model, self.cfg.temperature
        )
        population = [
            self.response_to_individual(response, response_id) 
            for response_id, response in enumerate(responses)
        ]
        return population


    def _run_single_iteration(self) -> None:
        """Execute one iteration of the evolutionary process (ReEvo: no individual self-evolution).
        
        Steps:
        1. Selection
        2. Population inter-evolution (crossover with reflection)
        3. Long-term reflection (train mode only)
        4. Mutation (train mode only)
        """
        # Check if all individuals are invalid
        if all(not individual["exec_success"] for individual in self.population):
            raise RuntimeError(
                f"All individuals are invalid. Please check the stdout files in {os.getcwd()}."
            )
        
        # Selection: add elitist to population if not already present
        if self.elitist is None or self.elitist in self.population:
            population_to_select = self.population
        else:
            population_to_select = [self.elitist] + self.population
        
        selected_population = self.random_select(population_to_select)
        if selected_population is None:
            raise RuntimeError("Selection failed. Please check the population.")
        
        # Population inter-evolution: reflection + crossover
        population_inter_envoltion_reflection_tuple = self.population_inter_envoltion_reflection(
            selected_population
        )
        population_inter_envoltion_population = self.population_inter_envoltion(
            population_inter_envoltion_reflection_tuple
        )
        self.population = self.evaluate_population(
            population_inter_envoltion_population, self.case_num
        )
        self.update_iter()

        # Long-term reflection and mutation (train mode only)
        if self.mode == "train":
            # Long-term reflection: aggregate short-term insights
            self.long_term_reflection(population_inter_envoltion_reflection_tuple[0])
            
            # Mutation: mutate elitist individual
            mutated_population = self.mutate()
            evaluated_mutated_population = self.evaluate_population(
                mutated_population, self.case_num
            )
            self.population.extend(evaluated_mutated_population)
            
            # Update iteration
            self.update_iter()
        
        self.function_evals += 1


    def evolve(self) -> tuple[str, str]:
        """Main evolutionary loop.
        
        Train mode: Performs multiple iterations until max_fe is reached
        Test mode: Performs only one iteration on loaded trained rules
        
        ReEvo does NOT include individual self-evolution (unlike SeEvo).
        
        Returns:
            Tuple of (best_code, best_code_path)
        """
        if self.mode == "test":
            # Test mode: run single iteration
            logging.info("Test mode: Running single iteration...")
            self._run_single_iteration()
        else:
            # Train mode: normal multi-iteration loop
            while self.function_evals < self.cfg.max_fe:
                self._run_single_iteration()
        
        return self.best_code_overall, self.best_code_path_overall
