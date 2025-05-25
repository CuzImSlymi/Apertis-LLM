import os
import sys
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    # Configure logging
    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
        
    logging.basicConfig(**logging_config)

def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save metrics to file
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
        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"Metrics file {filepath} does not exist")
            return {}
            
        # Load metrics from file
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
        self.last_output = ""
        self.last_error = ""
        
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return the result."""
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(code)
            temp_file = f.name
            
        try:
            # Execute the code in a subprocess
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                
                # Limit output size
                if len(stdout) > self.max_output_size:
                    stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
                if len(stderr) > self.max_output_size:
                    stderr = stderr[:self.max_output_size] + "\n... [error truncated]"
                    
                self.last_output = stdout
                self.last_error = stderr
                
                return {
                    "success": process.returncode == 0,
                    "output": stdout,
                    "error": stderr,
                    "return_code": process.returncode
                }
                
            except subprocess.TimeoutExpired:
                process.kill()
                self.last_error = f"Execution timed out after {self.timeout} seconds"
                return {
                    "success": False,
                    "output": "",
                    "error": self.last_error,
                    "return_code": -1
                }
                
        except Exception as e:
            self.last_error = str(e)
            return {
                "success": False,
                "output": "",
                "error": self.last_error,
                "return_code": -1
            }
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def get_last_output(self) -> str:
        """Get the output from the last execution."""
        return self.last_output
    
    def get_last_error(self) -> str:
        """Get the error from the last execution."""
        return self.last_error

class RewardCalculator:
    """Calculates rewards for the AZR self-play loop."""
    
    def __init__(self, config: Dict[str, Any], reward_modules: Dict[str, Any]):
        self.config = config
        self.reward_modules = reward_modules
        
    def calculate_task_rewards(self, task_info: Dict[str, Any], task_validation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for task generation."""
        rewards = {}
        
        # Calculate learnability reward if module exists
        if "learnability" in self.reward_modules:
            rewards["learnability"] = self.reward_modules["learnability"].calculate(task_validation)
            
        # Calculate diversity reward if module exists
        if "diversity" in self.reward_modules:
            rewards["diversity"] = self.reward_modules["diversity"].calculate(
                task_info.get("task", ""),
                task_info.get("type", "")
            )
            
        # Calculate complexity reward if module exists
        if "complexity" in self.reward_modules:
            rewards["complexity"] = self.reward_modules["complexity"].calculate(task_validation)
            
        # Calculate total reward
        total_reward = sum(rewards.values())
        rewards["total"] = total_reward
        
        return rewards
    
    def calculate_solution_rewards(self, task_info: Dict[str, Any], solution_info: Dict[str, Any], solution_validation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate rewards for solution generation."""
        rewards = {}
        
        # Calculate accuracy reward if module exists
        if "accuracy" in self.reward_modules:
            rewards["accuracy"] = self.reward_modules["accuracy"].calculate(solution_validation)
            
        # Calculate total reward
        total_reward = sum(rewards.values())
        rewards["total"] = total_reward
        
        return rewards

class SelfPlayTracker:
    """Tracks progress and metrics during the AZR self-play loop."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.metrics = {
            "iterations": 0,
            "tasks_generated": 0,
            "tasks_valid": 0,
            "solutions_generated": 0,
            "solutions_valid": 0,
            "task_rewards": [],
            "solution_rewards": [],
            "task_types": {"abduction": 0, "deduction": 0, "induction": 0}
        }
        self.task_history = []
        self.solution_history = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def update_task_metrics(self, task_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        """Update metrics after task generation and validation."""
        self.metrics["tasks_generated"] += 1
        
        if validation.get("is_valid", False):
            self.metrics["tasks_valid"] += 1
            
        # Update task type counts
        task_type = task_info.get("type", "")
        if task_type in self.metrics["task_types"]:
            self.metrics["task_types"][task_type] += 1
            
        # Store rewards
        self.metrics["task_rewards"].append(rewards)
        
        # Add to task history
        self.task_history.append({
            "task": task_info,
            "validation": validation,
            "rewards": rewards
        })
        
        # Save task to file if configured
        if self.config.get("save_tasks", True):
            self._save_task(task_info, validation, rewards)
    
    def update_solution_metrics(self, task_info: Dict[str, Any], solution_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        """Update metrics after solution generation and validation."""
        self.metrics["solutions_generated"] += 1
        
        if validation.get("is_valid", False):
            self.metrics["solutions_valid"] += 1
            
        # Store rewards
        self.metrics["solution_rewards"].append(rewards)
        
        # Add to solution history
        self.solution_history.append({
            "task": task_info,
            "solution": solution_info,
            "validation": validation,
            "rewards": rewards
        })
        
        # Save solution to file if configured
        if self.config.get("save_solutions", True):
            self._save_solution(task_info, solution_info, validation, rewards)
    
    def complete_iteration(self):
        """Update metrics after completing an iteration."""
        self.metrics["iterations"] += 1
        
        # Save metrics to file
        self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get task history."""
        return self.task_history
    
    def get_solution_history(self) -> List[Dict[str, Any]]:
        """Get solution history."""
        return self.solution_history
    
    def _save_task(self, task_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        """Save task to file."""
        task_dir = os.path.join(self.output_dir, "tasks")
        os.makedirs(task_dir, exist_ok=True)
        
        task_data = {
            "task": task_info,
            "validation": validation,
            "rewards": rewards,
            "timestamp": self._get_timestamp()
        }
        
        filename = f"task_{self.metrics['tasks_generated']:06d}.json"
        filepath = os.path.join(task_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2)
    
    def _save_solution(self, task_info: Dict[str, Any], solution_info: Dict[str, Any], validation: Dict[str, Any], rewards: Dict[str, float]):
        """Save solution to file."""
        solution_dir = os.path.join(self.output_dir, "solutions")
        os.makedirs(solution_dir, exist_ok=True)
        
        solution_data = {
            "task": task_info,
            "solution": solution_info,
            "validation": validation,
            "rewards": rewards,
            "timestamp": self._get_timestamp()
        }
        
        filename = f"solution_{self.metrics['solutions_generated']:06d}.json"
        filepath = os.path.join(solution_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(solution_data, f, indent=2)
    
    def _save_metrics(self):
        """Save metrics to file."""
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        
        # Calculate summary statistics
        summary = {
            "iterations": self.metrics["iterations"],
            "tasks_generated": self.metrics["tasks_generated"],
            "tasks_valid": self.metrics["tasks_valid"],
            "task_valid_rate": self.metrics["tasks_valid"] / max(1, self.metrics["tasks_generated"]),
            "solutions_generated": self.metrics["solutions_generated"],
            "solutions_valid": self.metrics["solutions_valid"],
            "solution_valid_rate": self.metrics["solutions_valid"] / max(1, self.metrics["solutions_generated"]),
            "task_types": self.metrics["task_types"],
            "avg_task_reward": self._calculate_avg_reward(self.metrics["task_rewards"]),
            "avg_solution_reward": self._calculate_avg_reward(self.metrics["solution_rewards"]),
            "timestamp": self._get_timestamp()
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_avg_reward(self, rewards: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average rewards."""
        if not rewards:
            return {}
            
        # Initialize with keys from first reward dict
        avg_rewards = {key: 0.0 for key in rewards[0].keys()}
        
        # Sum rewards
        for reward in rewards:
            for key, value in reward.items():
                avg_rewards[key] += value
                
        # Calculate averages
        for key in avg_rewards:
            avg_rewards[key] /= len(rewards)
            
        return avg_rewards
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
