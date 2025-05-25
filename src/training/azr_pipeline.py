import os
import sys
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer # IMPORT THIS

from .azr.rewards import LearnabilityReward, DiversityReward, ComplexityReward
from .azr.data_construction import TaskGenerator, TaskValidator, SolutionGenerator, SolutionValidator
from .azr.utils import save_metrics, load_metrics, setup_logging, PythonExecutor

logger = logging.getLogger(__name__)

class AbsoluteZeroReasonerTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        self.model, self.hf_tokenizer = self._setup_model_and_tokenizer() # MODIFIED METHOD NAME
        
        self._init_components()
        
        self.metrics = {
            "iterations": 0,
            "tasks_generated": 0,
            "tasks_valid": 0,
            "solutions_generated": 0,
            "solutions_valid": 0,
            "task_rewards": [],
            "solution_rewards": [],
            "task_types": {"abduction": 0, "deduction": 0, "induction": 0},
            "validation_rates": {"tasks": [], "solutions": []}
        }
        
        if self.config.get("continue_from_checkpoint", False):
            checkpoint_path = self.config.get("checkpoint_path", "")
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.metrics = load_metrics(checkpoint_path)
                logger.info(f"Loaded metrics from checkpoint: {checkpoint_path}")
        
        logger.info("Initialized all AZR components")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def setup_logging(self):
        log_level = self.config.get("log_level", "INFO")
        log_file = self.config.get("log_file", None)
        setup_logging(log_level, log_file)
    
    def _setup_model_and_tokenizer(self) -> Tuple[torch.nn.Module, Any]: # RENAMED AND MODIFIED
        try:
            model_config_from_file = self.config.get("model", {}).copy()
            
            # --- Hugging Face Tokenizer Integration ---
            tokenizer_name = self.config.get("data", {}).get("tokenizer_name", "bert-base-uncased")
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info(f"Successfully loaded Hugging Face tokenizer: {tokenizer_name}")
                # Override vocab_size in model_config_from_file with the actual HF tokenizer vocab size
                model_config_from_file["vocab_size"] = hf_tokenizer.vocab_size
                # Set pad, bos, eos token IDs from HF tokenizer if available, else use ApertisConfig defaults
                if hf_tokenizer.pad_token_id is not None:
                    model_config_from_file["pad_token_id"] = hf_tokenizer.pad_token_id
                if hf_tokenizer.bos_token_id is not None:
                    model_config_from_file["bos_token_id"] = hf_tokenizer.bos_token_id
                if hf_tokenizer.eos_token_id is not None:
                    model_config_from_file["eos_token_id"] = hf_tokenizer.eos_token_id
                if hf_tokenizer.unk_token_id is not None:
                     model_config_from_file["unk_token_id"] = hf_tokenizer.unk_token_id

            except Exception as e:
                logger.error(f"Failed to load Hugging Face tokenizer '{tokenizer_name}'. Error: {e}", exc_info=True)
                logger.warning("Falling back to minimal tokenizer due to HF tokenizer load failure.")
                # Fallback minimal tokenizer (less ideal)
                minimal_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "[SEP]": 102, "[CLS]": 101} # Added common BERT specials
                class MinimalHFCompatTokenizer: # A mock object that has some HF tokenizer attributes
                    def __init__(self, vocab):
                        self.vocab = vocab
                        self.ids_to_tokens = {v: k for k,v in vocab.items()}
                        self.vocab_size = len(vocab)
                        self.pad_token_id = vocab.get("<pad>", 0)
                        self.bos_token_id = vocab.get("<bos>", 2)
                        self.eos_token_id = vocab.get("<eos>", 3)
                        self.unk_token_id = vocab.get("<unk>", 1)
                        self.sep_token_id = vocab.get("[SEP]", 102)
                        self.cls_token_id = vocab.get("[CLS]", 101)

                    def encode(self, text, add_special_tokens=True, truncation=True, max_length=512, return_tensors=None):
                        # Extremely basic tokenization
                        tokens = text.lower().split()
                        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
                        if add_special_tokens and hasattr(self, 'cls_token_id') and hasattr(self, 'sep_token_id'):
                            ids = [self.cls_token_id] + ids + [self.sep_token_id]
                        if truncation and len(ids) > max_length:
                            ids = ids[:max_length-1] + [self.sep_token_id] if add_special_tokens and hasattr(self, 'sep_token_id') else ids[:max_length]
                        if return_tensors == "pt": return {"input_ids": torch.tensor([ids])}
                        return ids

                    def decode(self, token_ids, skip_special_tokens=True):
                        tokens = []
                        for tid in token_ids:
                            token_str = self.ids_to_tokens.get(tid, f"<ID:{tid}>")
                            if skip_special_tokens and token_str in ["<pad>", "<bos>", "<eos>", "<unk>", "[CLS]", "[SEP]"]:
                                continue
                            tokens.append(token_str)
                        return " ".join(tokens)

                    def __call__(self, text, **kwargs): # Make it callable like HF tokenizers
                        return self.encode(text, **kwargs)

                hf_tokenizer = MinimalHFCompatTokenizer(minimal_vocab)
                model_config_from_file["vocab_size"] = hf_tokenizer.vocab_size
                model_config_from_file["pad_token_id"] = hf_tokenizer.pad_token_id
                model_config_from_file["bos_token_id"] = hf_tokenizer.bos_token_id
                model_config_from_file["eos_token_id"] = hf_tokenizer.eos_token_id
                model_config_from_file["unk_token_id"] = hf_tokenizer.unk_token_id


            # --- End Hugging Face Tokenizer Integration ---

            param_mapping = {
                "rms_norm_eps": "layer_norm_eps",
                "attention_dropout": "attention_probs_dropout_prob"
            }
            for old_param, new_param in param_mapping.items():
                if old_param in model_config_from_file:
                    model_config_from_file[new_param] = model_config_from_file.pop(old_param)
            
            from src.model.core import ApertisConfig, ApertisForCausalLM
            
            valid_params = set(ApertisConfig.__init__.__code__.co_varnames)
            invalid_params = []
            filtered_config_for_apertis = {}
            for param, value in model_config_from_file.items():
                if param in valid_params:
                    filtered_config_for_apertis[param] = value
                else:
                    invalid_params.append(param)
            
            if invalid_params:
                logger.info(f"Filtered out unsupported parameters for ApertisConfig: {', '.join(invalid_params)}")
            
            model_config_obj = ApertisConfig(**filtered_config_for_apertis)
            
            model = ApertisForCausalLM(model_config_obj)
            logger.info(f"Created ApertisForCausalLM with actual config: {model_config_obj.to_dict()}")
            
            device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"Using device: {device}")
            
            return model, hf_tokenizer # Return the HF tokenizer object
            
        except Exception as e:
            logger.error(f"Error creating model or tokenizer: {e}", exc_info=True)
            raise
    
    def _init_components(self):
        logger.info("Starting _init_components")
        azr_specific_config = self.config.get("azr", {})
        if not azr_specific_config:
            logger.error("AZR specific configuration ('azr' key) is missing or empty!")
        
        logger.info(f"AZR specific config keys found: {list(azr_specific_config.keys())}")

        python_executor_config = azr_specific_config.get("python_executor", {})
        logger.info(f"Python executor config: {python_executor_config}")
        try:
            self.python_executor = PythonExecutor(python_executor_config)
            logger.info("PythonExecutor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PythonExecutor: {e}", exc_info=True)
            self.python_executor = None 

        task_gen_conf = azr_specific_config.get("task_generator", {})
        logger.info(f"TaskGenerator using config: {task_gen_conf}")
        self.task_generator = TaskGenerator(task_gen_conf) # TaskGenerator will use self.hf_tokenizer internally
        logger.info("TaskGenerator initialized.")

        task_val_conf = azr_specific_config.get("task_validator", {})
        logger.info(f"TaskValidator using config: {task_val_conf}")
        self.task_validator = TaskValidator(config=task_val_conf, python_executor=self.python_executor)
        logger.info("TaskValidator initialized.")
        
        sol_gen_conf = azr_specific_config.get("solution_generator", {})
        logger.info(f"SolutionGenerator using config: {sol_gen_conf}")
        self.solution_generator = SolutionGenerator(sol_gen_conf) # SolutionGenerator will use self.hf_tokenizer
        logger.info("SolutionGenerator initialized.")

        sol_val_conf = azr_specific_config.get("solution_validator", {})
        logger.info(f"SolutionValidator using config: {sol_val_conf}")
        self.solution_validator = SolutionValidator(config=sol_val_conf, python_executor=self.python_executor)
        logger.info("SolutionValidator initialized.")
        
        learn_conf = azr_specific_config.get("learnability_reward", {})
        logger.info(f"LearnabilityReward using config: {learn_conf}")
        self.learnability_reward = LearnabilityReward(learn_conf)
        logger.info("LearnabilityReward initialized.")

        div_conf = azr_specific_config.get("diversity_reward", {})
        logger.info(f"DiversityReward using config: {div_conf}")
        self.diversity_reward = DiversityReward(div_conf)
        logger.info("DiversityReward initialized.")

        comp_conf = azr_specific_config.get("complexity_reward", {})
        logger.info(f"ComplexityReward using config: {comp_conf}")
        self.complexity_reward = ComplexityReward(comp_conf)
        logger.info("ComplexityReward initialized.")
        
        logger.info("Finished _init_components successfully.")

    def train(self):
        logger.info("Starting Absolute Zero Reasoner training")
        
        num_iterations = self.config.get("azr",{}).get("num_iterations", 100) # Get from azr config
        tasks_per_iteration = self.config.get("azr",{}).get("tasks_per_iteration", 5)
        checkpoint_interval = self.config.get("azr",{}).get("checkpoint_interval", 10)
        
        force_accept_tasks = self.config.get("azr",{}).get("force_accept_tasks", True)
        force_accept_solutions = self.config.get("azr",{}).get("force_accept_solutions", True)
        force_accept_threshold = self.config.get("azr",{}).get("force_accept_threshold", 10)
        min_valid_tasks_before_validation = self.config.get("azr",{}).get("min_valid_tasks_before_validation", 20)
        
        start_iteration = self.metrics["iterations"] + 1
        
        for iteration in range(start_iteration, start_iteration + num_iterations):
            logger.info(f"Starting iteration {iteration}/{start_iteration + num_iterations - 1}")
            
            should_force_accept_tasks = force_accept_tasks
            should_force_accept_solutions = force_accept_solutions
            
            if iteration > force_accept_threshold:
                should_force_accept_tasks = False
                logger.info(f"Iteration {iteration} > threshold {force_accept_threshold}: Disabling forced task acceptance")
                
            if self.metrics["tasks_valid"] > min_valid_tasks_before_validation:
                should_force_accept_solutions = False
                logger.info(f"Valid tasks ({self.metrics['tasks_valid']}) > threshold {min_valid_tasks_before_validation}: Disabling forced solution acceptance")
            
            valid_tasks = []
            task_rewards_iter = [] # Renamed to avoid conflict with attribute
            tasks_attempted = 0
            tasks_valid_this_iteration = 0
            
            for task_idx in range(1, tasks_per_iteration + 1):
                logger.info(f"Generating task {task_idx}/{tasks_per_iteration}")
                
                # Pass the Hugging Face tokenizer object
                task_info = self.task_generator.generate_task(self.model, self.hf_tokenizer) 
                self.metrics["tasks_generated"] += 1
                tasks_attempted += 1
                
                task_text = task_info.get("task", "")
                task_type = task_info.get("type", "")
                logger.info(f"Generated task {task_idx} (type: {task_type}):\n{task_text}")
                
                if task_type in self.metrics["task_types"]:
                    self.metrics["task_types"][task_type] += 1
                
                validation_result = self.task_validator.validate(task_info)
                is_valid = validation_result["is_valid"]
                if should_force_accept_tasks and not is_valid:
                    logger.info(f"Force accepting task {task_idx} to ensure training progress")
                    is_valid = True
                
                if is_valid:
                    logger.info(f"Task {task_idx} is valid, proceeding with solution generation")
                    valid_tasks.append(task_info)
                    self.metrics["tasks_valid"] += 1
                    tasks_valid_this_iteration += 1
                    
                    learnability = self.learnability_reward.calculate(task_info)
                    diversity = self.diversity_reward.calculate(task_info, valid_tasks[:-1])
                    complexity = self.complexity_reward.calculate(task_info)
                    
                    reward = {
                        "learnability": learnability, "diversity": diversity,
                        "complexity": complexity, "total": learnability + diversity + complexity
                    }
                    task_rewards_iter.append(reward) # Use local variable
                    self.metrics["task_rewards"].append(reward) # Append to class attribute list
                else:
                    logger.info(f"Task {task_idx} is invalid, skipping. Validation result: {validation_result}")
            
            task_validation_rate = tasks_valid_this_iteration / tasks_attempted if tasks_attempted > 0 else 0
            self.metrics["validation_rates"]["tasks"].append(task_validation_rate)
            logger.info(f"Task validation rate for iteration {iteration}: {task_validation_rate:.2f}")
            
            valid_solutions = []
            solution_rewards_iter = [] # Renamed
            solutions_attempted = 0
            solutions_valid_this_iteration = 0
            
            for sol_task_idx, task_info_for_sol in enumerate(valid_tasks): # Renamed task_info
                logger.info(f"Generating solution for task {sol_task_idx + 1}/{len(valid_tasks)}")
                
                # Pass the Hugging Face tokenizer object
                solution_info = self.solution_generator.generate_solution(task_info_for_sol, self.model, self.hf_tokenizer)
                self.metrics["solutions_generated"] += 1
                solutions_attempted += 1
                
                task_text_for_sol = task_info_for_sol.get("task", "") # Use renamed task_info
                solution_text = solution_info.get("solution", "")
                logger.info(f"Task: {task_text_for_sol}\nGenerated solution:\n{solution_text}")
                
                validation_result_sol = self.solution_validator.validate(task_info_for_sol, solution_info) # Use renamed task_info
                
                is_sol_valid = validation_result_sol["is_valid"] # Renamed
                if should_force_accept_solutions and not is_sol_valid:
                    logger.info(f"Force accepting solution for task {sol_task_idx + 1} to ensure training progress")
                    is_sol_valid = True
                
                if is_sol_valid:
                    logger.info(f"Solution for task {sol_task_idx + 1} is valid")
                    valid_solutions.append(solution_info)
                    self.metrics["solutions_valid"] += 1
                    solutions_valid_this_iteration += 1
                    
                    reward_sol = {"correctness": validation_result_sol.get("correctness", 0.5)} # Renamed
                    solution_rewards_iter.append(reward_sol) # Use local variable
                    self.metrics["solution_rewards"].append(reward_sol) # Append to class attribute list
                else:
                    logger.info(f"Solution for task {sol_task_idx + 1} is invalid, skipping. Validation result: {validation_result_sol}")
            
            solution_validation_rate = solutions_valid_this_iteration / solutions_attempted if solutions_attempted > 0 else 0
            self.metrics["validation_rates"]["solutions"].append(solution_validation_rate)
            logger.info(f"Solution validation rate for iteration {iteration}: {solution_validation_rate:.2f}")
            
            self.metrics["iterations"] = iteration
            logger.info(f"Iteration {iteration} metrics: {self.metrics}")
            
            if iteration % checkpoint_interval == 0:
                self._save_checkpoint(iteration)
        
        self._save_checkpoint(self.metrics["iterations"])
        logger.info("AZR training completed")
    
    def _save_checkpoint(self, iteration: int):
        checkpoint_dir = self.config.get("azr",{}).get("checkpoint_dir", "checkpoints") # Get from azr config
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        metrics_path = os.path.join(checkpoint_dir, f"metrics_iter_{iteration}.json")
        save_metrics(self.metrics, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
        model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}")
        self.model.save_pretrained(model_path) # Assumes ApertisForCausalLM has save_pretrained
        # If you also want to save the HF tokenizer config (though it's loaded by name):
        # self.hf_tokenizer.save_pretrained(model_path) 
        logger.info(f"Saved model to {model_path}")

def train_azr(config_path: str):
    trainer = AbsoluteZeroReasonerTrainer(config_path)
    trainer.train()

def train_from_config(config_path: str):
    return train_azr(config_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path_arg = sys.argv[1]
        train_azr(config_path_arg)
    else:
        print("Usage: python azr_pipeline.py <config_path>")