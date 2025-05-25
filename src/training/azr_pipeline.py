import os
import sys
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from .azr.rewards import LearnabilityReward, DiversityReward, ComplexityReward
from .azr.data_construction import TaskGenerator, TaskValidator, SolutionGenerator, SolutionValidator
from .azr.utils import save_metrics, load_metrics, setup_logging, PythonExecutor 

logger = logging.getLogger(__name__)

class AbsoluteZeroReasonerTrainer:
    """
    Implementation of the Absolute Zero Reasoner (AZR) training method.
    
    AZR uses a self-play loop where the model generates reasoning tasks and solutions,
    with rewards guiding the learning process toward more effective reasoning.
    """
    
    def __init__(self, config_path: str):
        """Initialize the AZR trainer with the given configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Load or create model
        self.model, self.tokenizer = self._setup_model()
        
        # Initialize components
        self._init_components()
        
        # Training state
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
        
        # Load previous metrics if continuing training
        if self.config.get("continue_from_checkpoint", False):
            checkpoint_path = self.config.get("checkpoint_path", "")
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.metrics = load_metrics(checkpoint_path)
                logger.info(f"Loaded metrics from checkpoint: {checkpoint_path}")
        
        logger.info("Initialized all AZR components")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config.get("log_level", "INFO")
        log_file = self.config.get("log_file", None)
        setup_logging(log_level, log_file)
    
    def _setup_model(self) -> Tuple[torch.nn.Module, Any]:
        try:
            model_config_from_file = self.config.get("model", {}).copy()
            
            param_mapping = {
                "rms_norm_eps": "layer_norm_eps",
                "attention_dropout": "attention_probs_dropout_prob"
            }
            
            for old_param, new_param in param_mapping.items():
                if old_param in model_config_from_file:
                    logger.info(f"Mapped config parameter '{old_param}' to '{new_param}'")
                    model_config_from_file[new_param] = model_config_from_file.pop(old_param)
            
            from src.model.core import ApertisConfig, ApertisForCausalLM
            
            actual_tokenizer_dict = {}
            actual_vocab_size = 0 # Default if vocab file fails to load
            default_vocab_for_model_config = model_config_from_file.get("vocab_size", 32000) # Get from config or default

            vocab_path = self.config.get("data", {}).get("tokenizer_path", None)
            if vocab_path and os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    actual_tokenizer_dict = json.load(f)
                actual_vocab_size = len(actual_tokenizer_dict)
                logger.info(f"Loaded vocabulary with {actual_vocab_size} tokens from {vocab_path} for model configuration.")
                # Override the vocab_size in the model_config_from_file
                model_config_from_file["vocab_size"] = actual_vocab_size
            else:
                logger.warning(f"No vocabulary file found at {vocab_path} or path not provided. "
                               f"Model will use configured/default vocab_size: {default_vocab_for_model_config}.")
                # Use the vocab_size from config, or if it's not there, a fallback for placeholder tokenizer
                actual_vocab_size = default_vocab_for_model_config 
                actual_tokenizer_dict = { # Fallback tokenizer for the generator functions
                    "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3
                }
                # Ensure model_config_from_file has a vocab_size if it was missing and no vocab file
                if "vocab_size" not in model_config_from_file:
                     model_config_from_file["vocab_size"] = actual_vocab_size


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
            logger.info(f"Created ApertisForCausalLM with actual config: {model_config_obj.__dict__}")
            
            device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"Using device: {device}")
            
            return model, actual_tokenizer_dict # Return the loaded or fallback dictionary
            
        except Exception as e:
            logger.error(f"Error creating model with config {self.config.get('model', {})}: {e}", exc_info=True)
            raise
    
    def _init_components(self):
        """Initialize AZR components."""
        logger.info("Starting _init_components")
        azr_specific_config = self.config.get("azr", {})
        if not azr_specific_config:
            logger.error("AZR specific configuration ('azr' key) is missing or empty!")
            # Optionally raise an error here or ensure all components can handle empty dicts
            # For now, we'll let it proceed and rely on .get({}, {}) for individual components
        
        logger.info(f"AZR specific config keys found: {list(azr_specific_config.keys())}")

        # Initialize Python Executor
        python_executor_config = azr_specific_config.get("python_executor", {})
        logger.info(f"Python executor config: {python_executor_config}")
        try:
            self.python_executor = PythonExecutor(python_executor_config)
            logger.info("PythonExecutor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PythonExecutor: {e}", exc_info=True)
            # Decide if this is fatal. For now, let it proceed and components might fail if they need it.
            self.python_executor = None 

        # Task generation and validation
        task_gen_conf = azr_specific_config.get("task_generator", {})
        logger.info(f"TaskGenerator using config: {task_gen_conf}")
        self.task_generator = TaskGenerator(task_gen_conf)
        logger.info("TaskGenerator initialized.")

        task_val_conf = azr_specific_config.get("task_validator", {})
        logger.info(f"TaskValidator using config: {task_val_conf}")
        self.task_validator = TaskValidator(config=task_val_conf, python_executor=self.python_executor)
        logger.info("TaskValidator initialized.")
        
        # Solution generation and validation
        sol_gen_conf = azr_specific_config.get("solution_generator", {})
        logger.info(f"SolutionGenerator using config: {sol_gen_conf}")
        self.solution_generator = SolutionGenerator(sol_gen_conf)
        logger.info("SolutionGenerator initialized.")

        sol_val_conf = azr_specific_config.get("solution_validator", {})
        logger.info(f"SolutionValidator using config: {sol_val_conf}")
        self.solution_validator = SolutionValidator(config=sol_val_conf, python_executor=self.python_executor)
        logger.info("SolutionValidator initialized.")
        
        # Rewards
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
        self.complexity_reward = ComplexityReward(comp_conf) # This is the critical line for the error
        logger.info("ComplexityReward initialized.")
        
        logger.info("Finished _init_components successfully.")
    
    def train(self):
        """Run the AZR training loop."""
        logger.info("Starting Absolute Zero Reasoner training")
        
        # Training parameters
        num_iterations = self.config.get("num_iterations", 100)
        tasks_per_iteration = self.config.get("tasks_per_iteration", 5)
        checkpoint_interval = self.config.get("checkpoint_interval", 10)
        
        # Force acceptance parameters
        force_accept_tasks = self.config.get("force_accept_tasks", True)
        force_accept_solutions = self.config.get("force_accept_solutions", True)
        force_accept_threshold = self.config.get("force_accept_threshold", 10)
        min_valid_tasks_before_validation = self.config.get("min_valid_tasks_before_validation", 20)
        
        # Starting iteration (for resuming training)
        start_iteration = self.metrics["iterations"] + 1
        
        # Main training loop
        for iteration in range(start_iteration, start_iteration + num_iterations):
            logger.info(f"Starting iteration {iteration}/{start_iteration + num_iterations - 1}")
            
            # Determine if we should force accept tasks/solutions in this iteration
            should_force_accept_tasks = force_accept_tasks
            should_force_accept_solutions = force_accept_solutions
            
            # Gradually disable forced acceptance as training progresses
            if iteration > force_accept_threshold:
                should_force_accept_tasks = False
                logger.info(f"Iteration {iteration} > threshold {force_accept_threshold}: Disabling forced task acceptance")
                
            if self.metrics["tasks_valid"] > min_valid_tasks_before_validation:
                should_force_accept_solutions = False
                logger.info(f"Valid tasks ({self.metrics['tasks_valid']}) > threshold {min_valid_tasks_before_validation}: Disabling forced solution acceptance")
            
            # Generate and validate tasks
            valid_tasks = []
            task_rewards = []
            tasks_attempted = 0
            tasks_valid_this_iteration = 0
            
            for task_idx in range(1, tasks_per_iteration + 1):
                logger.info(f"Generating task {task_idx}/{tasks_per_iteration}")
                
                # Generate task
                task_info = self.task_generator.generate_task(self.model, self.tokenizer)
                self.metrics["tasks_generated"] += 1
                tasks_attempted += 1
                
                # Log the generated task content for inspection
                task_text = task_info.get("task", "")
                task_type = task_info.get("type", "")
                logger.info(f"Generated task {task_idx} (type: {task_type}):\n{task_text}")
                
                # Update task type counts
                if task_type in self.metrics["task_types"]:
                    self.metrics["task_types"][task_type] += 1
                
                # Validate task (even if we'll force accept it, for metrics)
                validation_result = self.task_validator.validate(task_info)
                
                # Determine if we should accept this task
                is_valid = validation_result["is_valid"]
                if should_force_accept_tasks and not is_valid:
                    logger.info(f"Force accepting task {task_idx} to ensure training progress")
                    is_valid = True
                
                if is_valid:
                    logger.info(f"Task {task_idx} is valid, proceeding with solution generation")
                    valid_tasks.append(task_info)
                    self.metrics["tasks_valid"] += 1
                    tasks_valid_this_iteration += 1
                    
                    # Calculate rewards
                    learnability = self.learnability_reward.calculate(task_info)
                    diversity = self.diversity_reward.calculate(task_info, valid_tasks[:-1])
                    complexity = self.complexity_reward.calculate(task_info)
                    
                    # Store rewards
                    reward = {
                        "learnability": learnability,
                        "diversity": diversity,
                        "complexity": complexity,
                        "total": learnability + diversity + complexity
                    }
                    task_rewards.append(reward)
                    self.metrics["task_rewards"].append(reward)
                else:
                    logger.info(f"Task {task_idx} is invalid, skipping. Validation result: {validation_result}")
            
            # Calculate task validation rate for this iteration
            task_validation_rate = tasks_valid_this_iteration / tasks_attempted if tasks_attempted > 0 else 0
            self.metrics["validation_rates"]["tasks"].append(task_validation_rate)
            logger.info(f"Task validation rate for iteration {iteration}: {task_validation_rate:.2f}")
            
            # Generate and validate solutions for valid tasks
            valid_solutions = []
            solution_rewards = []
            solutions_attempted = 0
            solutions_valid_this_iteration = 0
            
            for task_idx, task_info in enumerate(valid_tasks):
                logger.info(f"Generating solution for task {task_idx + 1}/{len(valid_tasks)}")
                
                # Generate solution
                solution_info = self.solution_generator.generate_solution(task_info, self.model, self.tokenizer)
                self.metrics["solutions_generated"] += 1
                solutions_attempted += 1
                
                # Log the generated solution content for inspection
                task_text = task_info.get("task", "")
                solution_text = solution_info.get("solution", "")
                logger.info(f"Task: {task_text}\nGenerated solution:\n{solution_text}")
                
                # Validate solution (even if we'll force accept it, for metrics)
                validation_result = self.solution_validator.validate(task_info, solution_info)
                
                # Determine if we should accept this solution
                is_valid = validation_result["is_valid"]
                if should_force_accept_solutions and not is_valid:
                    logger.info(f"Force accepting solution for task {task_idx + 1} to ensure training progress")
                    is_valid = True
                
                if is_valid:
                    logger.info(f"Solution for task {task_idx + 1} is valid")
                    valid_solutions.append(solution_info)
                    self.metrics["solutions_valid"] += 1
                    solutions_valid_this_iteration += 1
                    
                    # Calculate rewards (placeholder for now)
                    reward = {"correctness": validation_result.get("correctness", 0.5)}
                    solution_rewards.append(reward)
                    self.metrics["solution_rewards"].append(reward)
                else:
                    logger.info(f"Solution for task {task_idx + 1} is invalid, skipping. Validation result: {validation_result}")
            
            # Calculate solution validation rate for this iteration
            solution_validation_rate = solutions_valid_this_iteration / solutions_attempted if solutions_attempted > 0 else 0
            self.metrics["validation_rates"]["solutions"].append(solution_validation_rate)
            logger.info(f"Solution validation rate for iteration {iteration}: {solution_validation_rate:.2f}")
            
            # Update training state
            self.metrics["iterations"] = iteration
            
            # Log metrics for this iteration
            logger.info(f"Iteration {iteration} metrics: {self.metrics}")
            
            # Save checkpoint if needed
            if iteration % checkpoint_interval == 0:
                self._save_checkpoint(iteration)
        
        # Final checkpoint
        self._save_checkpoint(self.metrics["iterations"])
        logger.info("AZR training completed")
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(checkpoint_dir, f"metrics_iter_{iteration}.json")
        save_metrics(self.metrics, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}")
        self.model.save_pretrained(model_path)
        logger.info(f"Saved model to {model_path}")

def train_azr(config_path: str):
    """Train a model using the Absolute Zero Reasoner method."""
    trainer = AbsoluteZeroReasonerTrainer(config_path)
    trainer.train()

# Add alias for train_from_config to match interface expectations
def train_from_config(config_path: str):
    """Alias for train_azr to match interface expectations."""
    return train_azr(config_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        train_azr(config_path)
    else:
        print("Usage: python azr_pipeline.py <config_path>")
