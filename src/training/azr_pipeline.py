import os
import sys
import json
import logging
import torch
import inspect
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import threading

from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.core import ApertisConfig, ApertisForCausalLM
from .azr.rewards import (
    ClarityReward,
    ComplexityReward,
    DiversityReward,
    AccuracyReward,
    CoherenceReward,
    RelevanceReward,
    StructureReward
)
from .azr.data_construction import TaskGenerator, TaskValidator, SolutionGenerator, SolutionValidator
from .azr.utils import setup_logging as azr_setup_logging, PythonExecutor, RewardCalculator, SelfPlayTracker

logger = logging.getLogger(__name__)

class AbsoluteZeroReasonerTrainer:
    def __init__(self, config_path: str, stop_event: Optional[threading.Event] = None):
        self.config_data = self._load_config(config_path)
        self.azr_config = self.config_data.get("azr", {})
        self.model_file_config = self.config_data.get("model", {})
        self.data_file_config = self.config_data.get("data", {})
        self.training_file_config = self.config_data.get("training", {})
        self.output_dir = self.training_file_config.get("output_dir", "output")

        self.setup_logging()
        
        self.model, self.hf_tokenizer = self._setup_model_and_tokenizer()
        
        self._init_components()
        
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
        log_file = self.azr_config.get("log_file")
        if log_file:
            os.makedirs(self.output_dir, exist_ok=True)
            log_file = os.path.join(self.output_dir, log_file)
        azr_setup_logging(log_level, log_file)
    
    def _setup_model_and_tokenizer(self) -> Tuple[torch.nn.Module, Any]:
        try:
            tokenizer_name = self.data_file_config.get("tokenizer_name", "gpt2")
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Successfully loaded Hugging Face tokenizer for AZR: {tokenizer_name}")

            apertis_config_params = self.model_file_config.copy()
            apertis_config_params["vocab_size"] = hf_tokenizer.vocab_size
            apertis_config_params["pad_token_id"] = hf_tokenizer.pad_token_id
            apertis_config_params["bos_token_id"] = hf_tokenizer.bos_token_id
            apertis_config_params["eos_token_id"] = hf_tokenizer.eos_token_id
            
            valid_apertis_params = inspect.signature(ApertisConfig.__init__).parameters.keys()
            filtered_apertis_params = {k: v for k, v in apertis_config_params.items() if k in valid_apertis_params}

            model_config_obj = ApertisConfig(**filtered_apertis_params)
            model = ApertisForCausalLM(model_config_obj)
            logger.info(f"Created ApertisForCausalLM for AZR with config: {model_config_obj.to_dict()}")
            
            device_str = self.training_file_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_str)
            model.to(device)
            logger.info(f"AZR Model using device: {device}")
            
            return model, hf_tokenizer
            
        except Exception as e:
            logger.error(f"Error creating model or tokenizer for AZR: {e}", exc_info=True)
            raise
    
    def _init_components(self):
        logger.info("Initializing AZR components...")
        
        self.python_executor = PythonExecutor(self.azr_config.get("python_executor", {}))
        self.task_generator = TaskGenerator(self.azr_config.get("task_generator", {}))
        self.task_validator = TaskValidator(self.azr_config.get("task_validator", {}), self.python_executor)
        self.solution_generator = SolutionGenerator(self.azr_config.get("solution_generator", {}))
        self.solution_validator = SolutionValidator(self.azr_config.get("solution_validator", {}), self.python_executor)
        
        reward_configs = self.azr_config.get("rewards", {})
        reward_modules = {
            "clarity": ClarityReward(reward_configs.get("clarity", {})),
            "complexity": ComplexityReward(reward_configs.get("complexity", {})),
            "diversity": DiversityReward(reward_configs.get("diversity", {})),
            "accuracy": AccuracyReward(reward_configs.get("accuracy", {})),
            "coherence": CoherenceReward(reward_configs.get("coherence", {})),
            "relevance": RelevanceReward(reward_configs.get("relevance", {})),
            "structure": StructureReward(reward_configs.get("structure", {})),
        }
        
        self.reward_calculator = RewardCalculator(reward_configs, reward_modules)
        
        azr_output_dir = os.path.join(self.output_dir, "azr_data")
        self.tracker = SelfPlayTracker(self.azr_config, azr_output_dir)
        
        logger.info("Finished AZR component initialization.")

    def train(self):
        logger.info("Starting Absolute Zero Reasoner training loop")
        
        num_iterations = self.azr_config.get("num_iterations", 100)
        tasks_per_iteration = self.azr_config.get("tasks_per_iteration", 5)
        checkpoint_interval = self.azr_config.get("checkpoint_interval", 10)
        
        force_accept_tasks_init = self.azr_config.get("force_accept_tasks", True)
        force_accept_solutions_init = self.azr_config.get("force_accept_solutions", True)
        force_accept_threshold = self.azr_config.get("force_accept_threshold", 10)
        min_valid_tasks_for_sol_val = self.azr_config.get("min_valid_tasks_before_validation", 20)
        
        start_iteration = self.tracker.get_metrics()["iterations"] + 1
        
        for iteration in range(start_iteration, start_iteration + num_iterations):
            if self.stop_event.is_set():
                logger.info(f"Stop event received. Halting AZR training at iteration {iteration}.")
                break
            
            logger.info(f"AZR Starting iteration {iteration}/{start_iteration + num_iterations - 1}")
            
            current_metrics = self.tracker.get_metrics()
            
            should_force_accept_tasks = force_accept_tasks_init and iteration <= force_accept_threshold
            should_force_accept_solutions = force_accept_solutions_init and current_metrics["tasks_valid"] <= min_valid_tasks_for_sol_val
            
            valid_tasks_for_this_iteration = []
            tasks_attempted_this_iter, tasks_valid_this_iter = 0, 0
            
            pbar_tasks = tqdm(range(tasks_per_iteration), desc=f"Iter {iteration} Tasks", disable=not logger.isEnabledFor(logging.INFO))
            for task_idx in pbar_tasks:
                if self.stop_event.is_set(): break
                
                task_info = self.task_generator.generate_task(self.model, self.hf_tokenizer) 
                tasks_attempted_this_iter += 1
                
                validation_result = self.task_validator.validate(task_info)
                is_valid = validation_result.get("is_valid", False)
                
                if should_force_accept_tasks and not is_valid:
                    is_valid = True
                    validation_result["is_valid"] = True
                    validation_result["reason"] = "Forced accept"
                
                if is_valid:
                    tasks_valid_this_iter += 1
                    task_rewards = self.reward_calculator.calculate_task_rewards(
                        task_info, validation_result, valid_tasks_for_this_iteration
                    )
                    self.tracker.update_task_metrics(task_info, validation_result, task_rewards)
                    valid_tasks_for_this_iteration.append(task_info)
                else:
                    self.tracker.update_task_metrics(task_info, validation_result, {"total": 0})
            
            if self.stop_event.is_set(): break
            
            task_val_rate = tasks_valid_this_iter / tasks_attempted_this_iter if tasks_attempted_this_iter > 0 else 0
            
            solutions_attempted_this_iter, solutions_valid_this_iter = 0, 0
            
            pbar_solutions = tqdm(valid_tasks_for_this_iteration, desc=f"Iter {iteration} Solutions", disable=not logger.isEnabledFor(logging.INFO))
            for current_task_info in pbar_solutions:
                if self.stop_event.is_set(): break

                solution_info = self.solution_generator.generate_solution(current_task_info, self.model, self.hf_tokenizer)
                solutions_attempted_this_iter += 1
                
                sol_validation_result = self.solution_validator.validate(current_task_info, solution_info)
                is_sol_valid = sol_validation_result.get("is_valid", False)
                
                if should_force_accept_solutions and not is_sol_valid:
                    is_sol_valid = True
                    sol_validation_result["is_valid"] = True
                    sol_validation_result["reason"] = "Forced accept"

                if is_sol_valid:
                    solutions_valid_this_iter += 1
                    solution_rewards = self.reward_calculator.calculate_solution_rewards(sol_validation_result)
                    self.tracker.update_solution_metrics(current_task_info, solution_info, sol_validation_result, solution_rewards)
                else:
                    self.tracker.update_solution_metrics(current_task_info, solution_info, sol_validation_result, {"total": 0})

            if self.stop_event.is_set(): break
            
            sol_val_rate = solutions_valid_this_iter / solutions_attempted_this_iter if solutions_attempted_this_iter > 0 else 0
            self.tracker.record_iteration_stats(task_val_rate, sol_val_rate)
            
            summary = self.tracker.get_summary_metrics()
            logger.info(f"Iteration {iteration} summary: Task Valid Rate={summary['task_valid_rate']:.2f}, Solution Valid Rate={summary['solution_valid_rate']:.2f}")
            logger.info(f"Avg Task Rewards: {summary['avg_task_rewards']}")
            logger.info(f"Avg Solution Rewards: {summary['avg_solution_rewards']}")
            
            if iteration % checkpoint_interval == 0 and not self.stop_event.is_set():
                self._save_checkpoint(iteration)
        
        if not self.stop_event.is_set():
            self._save_checkpoint(self.tracker.get_metrics()["iterations"])
        
        logger.info("AZR training process finished.")
        return self.tracker.get_summary_metrics()
    
    def _save_checkpoint(self, iteration: int):
        checkpoint_dir = self.azr_config.get("checkpoint_dir", "azr_checkpoints")
        full_checkpoint_dir = os.path.join(self.output_dir, checkpoint_dir)
        os.makedirs(full_checkpoint_dir, exist_ok=True)
        
        model_save_path = os.path.join(full_checkpoint_dir, f"model_iter_{iteration}")
        try:
            self.model.save_pretrained(model_save_path)
            self.hf_tokenizer.save_pretrained(model_save_path)
            logger.info(f"AZR Saved model and tokenizer to {model_save_path}")
        except Exception as e:
            logger.error(f"AZR Error saving model/tokenizer checkpoint: {e}", exc_info=True)


def train_from_config(config_path: str, stop_event: Optional[threading.Event] = None):
    trainer = AbsoluteZeroReasonerTrainer(config_path, stop_event)
    return trainer.train()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path_arg = sys.argv[1]
        train_from_config(config_path_arg)
    else:
        print("Usage: python -m src.training.azr_pipeline <path_to_azr_config.json>")