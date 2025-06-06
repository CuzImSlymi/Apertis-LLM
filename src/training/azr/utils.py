import os
import sys
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'handlers': [logging.StreamHandler(sys.stdout)]
    }
    
    if log_file:
        logging_config['handlers'].append(logging.FileHandler(log_file, mode='a'))
        
    logging.basicConfig(**logging_config)

def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving metrics to {filepath}: {e}")
        return False

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Metrics file {filepath} does not exist")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {filepath}")
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics from {filepath}: {e}")
        return {}

class PythonExecutor:
    """Executes Python code in a safe environment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get("timeout", 5)
        self.max_output_size = config.get("max_output_size", 10000)
        
    def execute(self, code: str) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
            
        try:
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                if len(stdout) > self.max_output_size:
                    stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
                if len(stderr) > self.max_output_size:
                    stderr = stderr[:self.max_output_size] + "\n... [error truncated]"
                    
                return {
                    "success": process.returncode == 0,
                    "output": stdout,
                    "error": stderr,
                    "return_code": process.returncode
                }
            except subprocess.TimeoutExpired:
                process.kill()
                error_msg = f"Execution timed out after {self.timeout} seconds"
                return {"success": False, "output": "", "error": error_msg, "return_code": -1}
                
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "return_code": -1}
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class RewardCalculator:
    def __init__(self, config: Dict[str, Any], reward_modules: Dict[str, Any]):
        self.config = config
        self.reward_modules = reward_modules
        self.task_reward_keys = self.config.get("task_reward_keys", ["clarity", "complexity", "diversity"])
        self.solution_reward_keys = self.config.get("solution_reward_keys", ["accuracy", "coherence", "relevance", "structure"])
        
    def calculate_task_rewards(self, task_info: Dict[str, Any], validation_result: Dict[str, Any], task_history: List[Dict[str, Any]]) -> Dict[str, float]:
        rewards = {}
        for key in self.task_reward_keys:
            if key in self.reward_modules:
                module = self.reward_modules[key]
                try:
                    if key == "diversity":
                        rewards[key] = module.calculate(task_info, task_history)
                    else:
                        rewards[key] = module.calculate(validation_result)
                except Exception as e:
                    logger.error(f"Error calculating task reward for '{key}': {e}", exc_info=True)
                    rewards[key] = 0.0
            
        rewards["total"] = sum(rewards.values())
        return rewards
    
    def calculate_solution_rewards(self, solution_validation: Dict[str, Any]) -> Dict[str, float]:
        rewards = {}
        for key in self.solution_reward_keys:
            if key in self.reward_modules:
                module = self.reward_modules[key]
                try:
                    rewards[key] = module.calculate(solution_validation)
                except Exception as e:
                    logger.error(f"Error calculating solution reward for '{key}': {e}", exc_info=True)
                    rewards[key] = 0.0
            
        rewards["total"] = sum(rewards.values())
        return rewards

class SelfPlayTracker:
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = self._load_or_initialize_metrics()
        self.task_history = []
        self.solution_history = []
        
    def _load_or_initialize_metrics(self) -> Dict[str, Any]:
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        if self.config.get("continue_from_checkpoint", False) and os.path.exists(metrics_file):
            logger.info(f"Continuing from existing metrics file: {metrics_file}")
            return load_metrics(metrics_file)

        return {
            "iterations": 0,
            "tasks_generated": 0,
            "tasks_valid": 0,
            "solutions_generated": 0,
            "solutions_valid": 0,
            "task_rewards": [],
            "solution_rewards": [],
            "task_types": defaultdict(int),
            "validation_rates": {
                "tasks": [],
                "solutions": []
            }
        }

    def update_task_metrics(self, task_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        self.metrics["tasks_generated"] += 1
        if validation.get("is_valid", False):
            self.metrics["tasks_valid"] += 1
            
        task_type = task_info.get("type", "unknown")
        self.metrics["task_types"][task_type] += 1
        self.metrics["task_rewards"].append(rewards)
        
        task_record = {"task": task_info, "validation": validation, "rewards": rewards}
        self.task_history.append(task_record)
        
        if self.config.get("save_tasks", True):
            self._save_record(task_record, "tasks", f"task_{self.metrics['tasks_generated']:06d}.json")
    
    def update_solution_metrics(self, task_info: Dict[str, Any], solution_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        self.metrics["solutions_generated"] += 1
        if validation.get("is_valid", False):
            self.metrics["solutions_valid"] += 1
            
        self.metrics["solution_rewards"].append(rewards)
        
        solution_record = {"task": task_info, "solution": solution_info, "validation": validation, "rewards": rewards}
        self.solution_history.append(solution_record)

        if self.config.get("save_solutions", True):
            self._save_record(solution_record, "solutions", f"solution_{self.metrics['solutions_generated']:06d}.json")
    
    def record_iteration_stats(self, task_val_rate: float, sol_val_rate: float):
        self.metrics["iterations"] += 1
        self.metrics["validation_rates"]["tasks"].append(task_val_rate)
        self.metrics["validation_rates"]["solutions"].append(sol_val_rate)
        self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        return self.task_history
    
    def _save_record(self, data: Dict[str, Any], subdir: str, filename: str):
        record_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(record_dir, exist_ok=True)
        filepath = os.path.join(record_dir, filename)
        
        record_data = data.copy()
        record_data["timestamp"] = self._get_timestamp()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, indent=2)

    def _save_metrics(self):
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        
        summary = self.get_summary_metrics()
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    def get_summary_metrics(self) -> Dict[str, Any]:
        tasks_gen = self.metrics.get("tasks_generated", 0)
        tasks_val = self.metrics.get("tasks_valid", 0)
        sols_gen = self.metrics.get("solutions_generated", 0)
        sols_val = self.metrics.get("solutions_valid", 0)

        return {
            "iterations": self.metrics.get("iterations", 0),
            "tasks_generated": tasks_gen,
            "tasks_valid": tasks_val,
            "task_valid_rate": tasks_val / max(1, tasks_gen),
            "solutions_generated": sols_gen,
            "solutions_valid": sols_val,
            "solution_valid_rate": sols_val / max(1, sols_gen),
            "task_types": self.metrics.get("task_types", {}),
            "avg_task_rewards": self._calculate_avg_reward(self.metrics.get("task_rewards", [])),
            "avg_solution_rewards": self._calculate_avg_reward(self.metrics.get("solution_rewards", [])),
            "avg_validation_rates": {
                "tasks": sum(self.metrics["validation_rates"]["tasks"]) / len(self.metrics["validation_rates"]["tasks"]) if self.metrics["validation_rates"]["tasks"] else 0,
                "solutions": sum(self.metrics["validation_rates"]["solutions"]) / len(self.metrics["validation_rates"]["solutions"]) if self.metrics["validation_rates"]["solutions"] else 0
            },
            "timestamp": self._get_timestamp()
        }

    def _calculate_avg_reward(self, rewards_history: List[Dict[str, float]]) -> Dict[str, float]:
        if not rewards_history:
            return {}
            
        avg_rewards = defaultdict(float)
        reward_counts = defaultdict(int)

        for reward_dict in rewards_history:
            for key, value in reward_dict.items():
                avg_rewards[key] += value
                reward_counts[key] += 1
                
        for key in avg_rewards:
            avg_rewards[key] /= reward_counts[key]
            
        return dict(avg_rewards)
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()
