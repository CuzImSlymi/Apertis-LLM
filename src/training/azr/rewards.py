import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class BaseReward:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def calculate(self, *args, **kwargs) -> float:
        raise NotImplementedError("Subclasses must implement calculate method")

class LearnabilityReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.min_threshold = config.get("min_threshold", 0.0)
        self.max_threshold = config.get("max_threshold", 1.0)
        
    def calculate(self, task_info: Dict[str, Any]) -> float:
        """Calculate learnability reward based on task information.
        
        Args:
            task_info: Dictionary containing task information
            
        Returns:
            float: Learnability reward score
        """
        # For now, return a simple fixed score
        # This can be enhanced later with more sophisticated metrics
        return 0.5 * self.weight

class AccuracyReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weight = config.get("weight", 1.0)
        self.partial_credit = config.get("partial_credit", True)
        
    def calculate(self, solution_validation_result: Dict[str, Any]) -> float:
        """Calculate accuracy reward based on solution validation results.
        
        Args:
            solution_validation_result: Dictionary containing validation results
            
        Returns:
            float: Accuracy reward score
        """
        if not solution_validation_result.get("is_valid", False):
            return 0.0
            
        # Extract validation metrics
        correctness = solution_validation_result.get("correctness", 0.0)
        
        if self.partial_credit:
            # Allow partial credit for partially correct solutions
            score = correctness
        else:
            # Binary reward - only fully correct solutions get reward
            score = 1.0 if correctness > 0.95 else 0.0
            
        # Apply weight
        return score * self.weight

class DiversityReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weight = config.get("weight", 0.5)
        self.history_size = config.get("history_size", 10)
        self.task_history = []
        
    def calculate(self, task_info: Union[Dict[str, Any], str], previous_tasks: Optional[List[Dict[str, Any]]] = None) -> float:
        """Calculate diversity reward based on task history.
        
        Args:
            task_info: The current task info (dict) or task text (str)
            previous_tasks: Optional list of previous tasks
            
        Returns:
            float: Diversity reward score
        """
        # Extract task text and type from task_info if it's a dictionary
        if isinstance(task_info, dict):
            task_text = task_info.get("task", "")
            task_type = task_info.get("type", "")
        else:
            # For backward compatibility
            task_text = task_info
            task_type = "unknown"
            
        # Simple n-gram based diversity calculation
        if not previous_tasks and not self.task_history:
            if previous_tasks is not None:
                # If empty list was explicitly provided
                return self.weight  # Max reward for first task
            # Otherwise use internal history
            self.task_history.append((task_text, task_type))
            return self.weight  # Max reward for first task
            
        # Use provided previous tasks if available, otherwise use internal history
        history = []
        if previous_tasks is not None:
            for prev_task_info in previous_tasks:
                if isinstance(prev_task_info, dict):
                    prev_text = prev_task_info.get("task", "")
                    prev_type = prev_task_info.get("type", "")
                    history.append((prev_text, prev_type))
                else:
                    # Handle case where previous tasks might be strings
                    history.append((prev_task_info, "unknown"))
        else:
            history = self.task_history
            
        # Calculate similarity to previous tasks
        similarities = []
        for prev_text, prev_type in history:
            # Type similarity (0.3 if same type, 0 if different)
            type_sim = 0.3 if prev_type == task_type else 0.0
            
            # Content similarity based on token overlap (crude approximation)
            # Handle empty strings or non-string inputs safely
            if not isinstance(task_text, str) or not isinstance(prev_text, str):
                content_sim = 0.0
            else:
                try:
                    tokens1 = set(task_text.lower().split())
                    tokens2 = set(prev_text.lower().split())
                    if not tokens1 or not tokens2:
                        content_sim = 0.0
                    else:
                        intersection = tokens1.intersection(tokens2)
                        union = tokens1.union(tokens2)
                        content_sim = 0.7 * (len(intersection) / len(union))
                except (AttributeError, TypeError):
                    # Handle any unexpected errors
                    logger.warning(f"Error calculating content similarity between '{task_text}' and '{prev_text}'")
                    content_sim = 0.0
                
            similarities.append(type_sim + content_sim)
            
        # Update internal history if not using provided previous tasks
        if previous_tasks is None:
            self.task_history.append((task_text, task_type))
            if len(self.task_history) > self.history_size:
                self.task_history.pop(0)
            
        # Diversity is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        diversity = 1.0 - max_similarity
        
        return diversity * self.weight

class ComplexityReward(BaseReward):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.weight = config.get("weight", 0.3)
        self.target_complexity = config.get("target_complexity", 0.7)
        self.tolerance = config.get("tolerance", 0.2)
        
    def calculate(self, task_info: Dict[str, Any]) -> float:
        """Calculate complexity reward based on task information.
        
        Args:
            task_info: Dictionary containing task information
            
        Returns:
            float: Complexity reward score
        """
        # For now, return a simple fixed score
        # This can be enhanced later with more sophisticated metrics
        return 0.5 * self.weight
