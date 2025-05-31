import os
import sys
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import threading

from transformers import AutoTokenizer # Ensure this is available

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # To find src
from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model
from .azr.rewards import LearnabilityReward, DiversityReward, ComplexityReward
from .azr.data_construction import TaskGenerator, TaskValidator, SolutionGenerator, SolutionValidator
from .azr.utils import save_metrics, load_metrics, setup_logging as azr_setup_logging, PythonExecutor

logger = logging.getLogger(__name__) # Will be configured by setup_logging

class AbsoluteZeroReasonerTrainer:
    def __init__(self, config_path: str, stop_event: Optional[threading.Event] = None):
        self.config_data = self._load_config(config_path) # Keep original loaded data
        self.azr_config = self.config_data.get("azr", {}) # Specific AZR settings
        self.model_file_config = self.config_data.get("model", {}) # Model structure from config file
        self.data_file_config = self.config_data.get("data", {}) # Data settings from config file
        self.training_file_config = self.config_data.get("training", {}) # Training settings

        self.setup_logging() # Use AZR's setup_logging utility
        
        self.model, self.hf_tokenizer = self._setup_model_and_tokenizer()
        
        self._init_components()
        
        self.metrics = {
            "iterations": 0,
            "tasks_generated": 0,
            "tasks_valid": 0,
            "solutions_generated": 0,
            "solutions_valid": 0,
            "task_rewards": [],
            "solution_rewards": [],
            "task_types": {task_type: 0 for task_type in self.azr_config.get("task_generator", {}).get("task_types", ["abduction", "deduction", "induction"])},
            "validation_rates": {"tasks": [], "solutions": []}
        }
        
        if self.azr_config.get("continue_from_checkpoint", False):
            checkpoint_path = self.azr_config.get("checkpoint_path", "")
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Assuming checkpoint_path is to a metrics file for now
                loaded_metrics = load_metrics(checkpoint_path) 
                if loaded_metrics: self.metrics.update(loaded_metrics)
                logger.info(f"Loaded metrics from checkpoint: {checkpoint_path}")
        
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        logger.info("Initialized all AZR components")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded AZR configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading AZR configuration from {config_path}: {e}")
            raise
    
    def setup_logging(self):
        log_level = self.azr_config.get("log_level", "INFO")
        log_file = self.azr_config.get("log_file", None) # If None, logs to console
        azr_setup_logging(log_level, log_file) # Use the utility from azr.utils
    
    def _setup_model_and_tokenizer(self) -> Tuple[torch.nn.Module, Any]:
        try:
            tokenizer_name_from_config = self.data_file_config.get("tokenizer_name", "bert-base-uncased")
            hf_tokenizer = None
            final_vocab_size = 0
            final_pad_id, final_bos_id, final_eos_id, final_unk_id = 0,1,2,3 # Defaults

            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_from_config)
                logger.info(f"Successfully loaded Hugging Face tokenizer for AZR: {tokenizer_name_from_config}")
                final_vocab_size = hf_tokenizer.vocab_size
                if hf_tokenizer.pad_token_id is not None: final_pad_id = hf_tokenizer.pad_token_id
                if hf_tokenizer.bos_token_id is not None: final_bos_id = hf_tokenizer.bos_token_id
                if hf_tokenizer.eos_token_id is not None: final_eos_id = hf_tokenizer.eos_token_id
                if hf_tokenizer.unk_token_id is not None: final_unk_id = hf_tokenizer.unk_token_id
            except Exception as e:
                logger.error(f"Failed to load Hugging Face tokenizer '{tokenizer_name_from_config}' for AZR. Error: {e}", exc_info=True)
                logger.warning("AZR training may fail or behave unexpectedly without a valid tokenizer.")
                # Proceeding without a tokenizer is risky for AZR, but let's try to allow it if user insists,
                # though task/solution generation will likely fail.
                # Set a fallback vocab size if model config doesn't specify one.
                final_vocab_size = self.model_file_config.get("vocab_size", 32000)


            # Prepare ApertisConfig parameters, ensuring they are valid
            apertis_config_params = self.model_file_config.copy() # Start with what's in the config file
            apertis_config_params["vocab_size"] = final_vocab_size
            apertis_config_params["pad_token_id"] = final_pad_id
            apertis_config_params["bos_token_id"] = final_bos_id
            apertis_config_params["eos_token_id"] = final_eos_id
            apertis_config_params["unk_token_id"] = final_unk_id

            # Filter for valid ApertisConfig args
            valid_apertis_params = inspect.signature(ApertisConfig.__init__).parameters.keys()
            filtered_apertis_params = {k: v for k, v in apertis_config_params.items() if k in valid_apertis_params}

            model_config_obj = ApertisConfig(**filtered_apertis_params)
            model = ApertisForCausalLM(model_config_obj)
            logger.info(f"Created ApertisForCausalLM for AZR with config: {model_config_obj.to_dict()}")
            
            device_str = self.training_file_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            # AZR UI passes specific GPU ID or "cpu"
            if device_str.startswith("cuda") and ":" not in device_str and torch.cuda.is_available():
                gpu_ids_from_config = self.training_file_config.get("gpu_ids")
                if gpu_ids_from_config and isinstance(gpu_ids_from_config, list) and len(gpu_ids_from_config) > 0:
                    device_str = f"cuda:{gpu_ids_from_config[0]}" # Use first specified GPU for AZR's single model
                else:
                    device_str = "cuda:0" # Default to GPU 0 if "cuda" specified but no ID
            
            device = torch.device(device_str)
            model.to(device)
            logger.info(f"AZR Model using device: {device}")
            
            return model, hf_tokenizer # hf_tokenizer might be None if loading failed
            
        except Exception as e:
            logger.error(f"Error creating model or tokenizer for AZR: {e}", exc_info=True)
            raise
    
    def _init_components(self):
        logger.info("Starting AZR _init_components")
        
        python_executor_config = self.azr_config.get("python_executor", {})
        self.python_executor = PythonExecutor(python_executor_config)
        logger.info("PythonExecutor initialized for AZR.")

        task_gen_conf = self.azr_config.get("task_generator", {})
        self.task_generator = TaskGenerator(task_gen_conf)
        logger.info("TaskGenerator initialized for AZR.")

        task_val_conf = self.azr_config.get("task_validator", {})
        self.task_validator = TaskValidator(config=task_val_conf, python_executor=self.python_executor)
        logger.info("TaskValidator initialized for AZR.")
        
        sol_gen_conf = self.azr_config.get("solution_generator", {})
        self.solution_generator = SolutionGenerator(sol_gen_conf)
        logger.info("SolutionGenerator initialized for AZR.")

        sol_val_conf = self.azr_config.get("solution_validator", {})
        self.solution_validator = SolutionValidator(config=sol_val_conf, python_executor=self.python_executor)
        logger.info("SolutionValidator initialized for AZR.")
        
        # Reward components
        self.learnability_reward = LearnabilityReward(self.azr_config.get("learnability_reward", {}))
        self.diversity_reward = DiversityReward(self.azr_config.get("diversity_reward", {}))
        self.complexity_reward = ComplexityReward(self.azr_config.get("complexity_reward", {}))
        # AccuracyReward is often part of SolutionValidator or called with its output
        
        logger.info("Finished AZR _init_components successfully.")

    def train(self):
        logger.info("Starting Absolute Zero Reasoner training loop")
        
        num_iterations = self.azr_config.get("num_iterations", 100)
        tasks_per_iteration = self.azr_config.get("tasks_per_iteration", 5)
        checkpoint_interval = self.azr_config.get("checkpoint_interval", 10)
        
        force_accept_tasks_init = self.azr_config.get("force_accept_tasks", True)
        force_accept_solutions_init = self.azr_config.get("force_accept_solutions", True)
        force_accept_threshold = self.azr_config.get("force_accept_threshold", 10)
        min_valid_tasks_for_sol_val = self.azr_config.get("min_valid_tasks_before_validation", 20)
        
        start_iteration = self.metrics["iterations"] + 1
        
        for iteration in range(start_iteration, start_iteration + num_iterations):
            if self.stop_event.is_set():
                logger.info(f"Stop event received. Halting AZR training at iteration {iteration}.")
                break
            logger.info(f"AZR Starting iteration {iteration}/{start_iteration + num_iterations - 1}")
            
            # Determine if forced acceptance should be active for this iteration
            should_force_accept_tasks_this_iter = force_accept_tasks_init
            if iteration > force_accept_threshold:
                should_force_accept_tasks_this_iter = False
                logger.info(f"AZR Iteration {iteration} > threshold {force_accept_threshold}: Disabling forced task acceptance.")
                
            should_force_accept_solutions_this_iter = force_accept_solutions_init
            if self.metrics["tasks_valid"] > min_valid_tasks_for_sol_val:
                should_force_accept_solutions_this_iter = False
                logger.info(f"AZR Valid tasks ({self.metrics['tasks_valid']}) > threshold {min_valid_tasks_for_sol_val}: Disabling forced solution acceptance.")
            
            valid_tasks_for_this_iteration = []
            tasks_attempted_this_iter, tasks_valid_this_iter = 0, 0
            
            for task_idx in range(1, tasks_per_iteration + 1):
                if self.stop_event.is_set(): break
                logger.info(f"AZR Generating task {task_idx}/{tasks_per_iteration} for iteration {iteration}")
                
                if not self.hf_tokenizer:
                    logger.error("AZR: No HF tokenizer available for task generation. Skipping iteration.")
                    break # Cannot proceed without tokenizer

                task_info = self.task_generator.generate_task(self.model, self.hf_tokenizer) 
                self.metrics["tasks_generated"] += 1
                tasks_attempted_this_iter += 1
                
                task_text = task_info.get("task", "")
                task_type = task_info.get("type", "unknown")
                logger.info(f"AZR Generated task {task_idx} (type: {task_type}):\n{task_text[:200]}...") # Log snippet
                
                if task_type in self.metrics["task_types"]: self.metrics["task_types"][task_type] += 1
                
                validation_result = self.task_validator.validate(task_info)
                is_valid = validation_result.get("is_valid", False)
                
                if should_force_accept_tasks_this_iter and not is_valid:
                    logger.info(f"AZR Force accepting task {task_idx} (Iter {iteration})")
                    is_valid = True
                
                if is_valid:
                    logger.info(f"AZR Task {task_idx} is valid.")
                    valid_tasks_for_this_iteration.append(task_info)
                    self.metrics["tasks_valid"] += 1
                    tasks_valid_this_iter += 1
                    
                    # Calculate rewards for the valid task
                    # Learnability reward might use validation_result or task_info
                    task_info['validation_result'] = validation_result # Add validation for reward calc
                    learn_reward = self.learnability_reward.calculate(task_info)
                    div_reward = self.diversity_reward.calculate(task_info, valid_tasks_for_this_iteration[:-1])
                    comp_reward = self.complexity_reward.calculate(task_info)
                    
                    current_task_rewards = {"learnability": learn_reward, "diversity": div_reward, "complexity": comp_reward}
                    current_task_rewards["total"] = sum(current_task_rewards.values())
                    self.metrics["task_rewards"].append(current_task_rewards)
                else:
                    logger.info(f"AZR Task {task_idx} invalid. Reason: {validation_result.get('reason', 'N/A')}")
            
            if self.stop_event.is_set(): break
            if not self.hf_tokenizer and tasks_per_iteration > 0 : break # Broke from inner loop due to no tokenizer

            task_val_rate = tasks_valid_this_iter / tasks_attempted_this_iter if tasks_attempted_this_iter > 0 else 0
            self.metrics["validation_rates"]["tasks"].append(task_val_rate)
            logger.info(f"AZR Task validation rate for iteration {iteration}: {task_val_rate:.2f}")
            
            solutions_attempted_this_iter, solutions_valid_this_iter = 0, 0
            
            for sol_idx, current_task_info in enumerate(valid_tasks_for_this_iteration):
                if self.stop_event.is_set(): break
                logger.info(f"AZR Generating solution for task {sol_idx + 1}/{len(valid_tasks_for_this_iteration)}")
                
                if not self.hf_tokenizer: # Should have been caught earlier
                    logger.error("AZR: No HF tokenizer available for solution generation. Skipping.")
                    break 

                solution_info = self.solution_generator.generate_solution(current_task_info, self.model, self.hf_tokenizer)
                self.metrics["solutions_generated"] += 1
                solutions_attempted_this_iter += 1
                
                logger.info(f"AZR Generated solution:\n{solution_info.get('solution', '')[:200]}...")
                
                sol_validation_result = self.solution_validator.validate(current_task_info, solution_info)
                is_sol_valid = sol_validation_result.get("is_valid", False)
                
                if should_force_accept_solutions_this_iter and not is_sol_valid:
                    logger.info(f"AZR Force accepting solution for task {sol_idx + 1} (Iter {iteration})")
                    is_sol_valid = True
                
                if is_sol_valid:
                    logger.info(f"AZR Solution for task {sol_idx + 1} is valid.")
                    self.metrics["solutions_valid"] += 1
                    solutions_valid_this_iter += 1
                    
                    # AccuracyReward is typically calculated based on solution_validation_result
                    # Let's assume AccuracyReward uses a 'correctness' score from sol_validation_result
                    acc_reward_val = sol_validation_result.get("correctness", 0.0) # Default if not present
                    
                    # For AZR, 'accuracy' might be less direct. We can use the validation 'correctness' as a proxy.
                    current_solution_rewards = {"accuracy_proxy": acc_reward_val} 
                    current_solution_rewards["total"] = acc_reward_val # Or sum if more solution rewards are added
                    self.metrics["solution_rewards"].append(current_solution_rewards)
                else:
                    logger.info(f"AZR Solution invalid. Reason: {sol_validation_result.get('reason', 'N/A')}")
            
            if self.stop_event.is_set(): break
            if not self.hf_tokenizer and len(valid_tasks_for_this_iteration) > 0: break

            sol_val_rate = solutions_valid_this_iter / solutions_attempted_this_iter if solutions_attempted_this_iter > 0 else 0
            self.metrics["validation_rates"]["solutions"].append(sol_val_rate)
            logger.info(f"AZR Solution validation rate for iteration {iteration}: {sol_val_rate:.2f}")
            
            self.metrics["iterations"] = iteration
            logger.info(f"AZR Iteration {iteration} completed. Current cumulative metrics: Tasks Valid: {self.metrics['tasks_valid']}, Solutions Valid: {self.metrics['solutions_valid']}")
            
            if iteration % checkpoint_interval == 0 and not self.stop_event.is_set():
                self._save_checkpoint(iteration)
        
        if not self.stop_event.is_set(): # Save final checkpoint if not stopped
            self._save_checkpoint(self.metrics["iterations"])
        else:
            logger.info("AZR training was stopped. Final checkpoint might not be saved unless it was due on this iteration.")

        logger.info("AZR training process finished.")
        return self.metrics # Return final metrics
    
    def _save_checkpoint(self, iteration: int):
        checkpoint_dir = self.azr_config.get("checkpoint_dir", "azr_checkpoints") # Default to specific AZR dir
        # Ensure output_dir from main training config is prepended if checkpoint_dir is relative
        main_output_dir = self.training_file_config.get("output_dir", "output")
        full_checkpoint_dir = os.path.join(main_output_dir, checkpoint_dir)

        os.makedirs(full_checkpoint_dir, exist_ok=True)
        
        metrics_path = os.path.join(full_checkpoint_dir, f"metrics_iter_{iteration}.json")
        save_metrics(self.metrics, metrics_path) # save_metrics handles dir creation
        logger.info(f"AZR Saved metrics to {metrics_path}")
        
        # Save model using ApertisForCausalLM's save_pretrained
        model_save_path = os.path.join(full_checkpoint_dir, f"model_iter_{iteration}")
        try:
            self.model.save_pretrained(model_save_path)
            logger.info(f"AZR Saved model to {model_save_path} using Apertis save_pretrained.")
            # If HF tokenizer was loaded, save it too
            if self.hf_tokenizer and hasattr(self.hf_tokenizer, "save_pretrained"):
                self.hf_tokenizer.save_pretrained(model_save_path)
                logger.info(f"AZR Saved HF tokenizer alongside model to {model_save_path}.")
        except Exception as e:
            logger.error(f"AZR Error saving model/tokenizer checkpoint: {e}", exc_info=True)


def train_from_config(config_path: str, stop_event: Optional[threading.Event] = None):
    import inspect # Ensure inspect is imported if used in _setup_model_and_tokenizer
    trainer = AbsoluteZeroReasonerTrainer(config_path, stop_event)
    return trainer.train() # Return final metrics

if __name__ == "__main__":
    import inspect # For ApertisConfig filtering in _setup_model_and_tokenizer
    if len(sys.argv) > 1:
        config_path_arg = sys.argv[1]
        train_from_config(config_path_arg)
    else:
        print("Usage: python azr_pipeline.py <path_to_azr_config.json>")